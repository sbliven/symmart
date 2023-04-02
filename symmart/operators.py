"""Manipulate symmetry groups algebraically
"""
import functools
import itertools
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp


class MatrixOperator:
    def __init__(self, mat, name=None):
        self.mat = sp.Matrix(mat)
        self.name = name

    def __mul__(self, other):
        return self @ other

    def __matmul__(self, other):
        if isinstance(other, MatrixOperator):
            if self.name and other.name:
                name = f"{self.name} * {other.name}"
            else:
                name = None
            return MatrixOperator(self.mat * other.mat, name=name)
        return self.mat * other

    def __eq__(self, other):
        if isinstance(other, MatrixOperator):
            return self.mat == other.mat
        return False

    def __hash__(self):
        return hash(tuple(self.mat.flat()))

    def __call__(self, z):
        return self.op * np.asarray(z)

    def __str__(self):
        return self.name or self.op_str()

    def op_str(self) -> str:
        "String as an operator, eg '[x, y]'"
        x, y = sp.symbols("x y")
        prod = self.mat @ sp.Matrix([[x], [y], [1]])
        return str(prod[:2])

    def __repr__(self):
        return f"MatrixOperator({self.mat.tolist()!r})"

    @functools.cache
    def fixed_point(self) -> Optional[List[sp.core.Number]]:
        "Return a stable point for this operator"
        eig = self.mat.eigenvects()
        pts = [v[:2] for λ, _, vs in eig if λ == 1 for v in vs if v[2] != 0]
        vec = [v[:2] for λ, _, vs in eig if λ == 1 for v in vs if v[2] == 0]
        return pts[0] if len(pts) == 1 and len(vec) == 0 else None

    @functools.cache
    def stable_line(
        self,
    ) -> Optional[
        Tuple[List[sp.core.Number], List[sp.core.Number], List[sp.core.Number]]
    ]:
        "Return stable lines as a (point, vector, translation) tuple"
        translation = (self.mat ** 2) / 2
        A = self.mat - translation + sp.eye(3)  # convert glide to mirror
        A /= A[2, 2]
        eig = A.eigenvects()
        pts = [v[:2] for λ, _, vs in eig if λ == 1 for v in vs if v[2] != 0]
        vec = [v[:2] for λ, _, vs in eig if λ == 1 for v in vs if v[2] == 0]
        if len(vec) != 1:
            return None
        assert len(pts) == 1, eig
        assert len(vec) == 1, eig

        return pts[0], vec[0], list(translation[:2, 2])

    def in_unit_cell(self):
        "Check if the stable point/line of this operation is inside the unit cell"
        # Point symmetry
        pt = self.fixed_point()
        if pt is not None:
            return 0 <= pt[0] < 1 and 0 <= pt[1] < 1
        # reflection/glide
        ln = self.stable_line()
        if ln is not None:
            (px, py), (lx, ly), (tx, ty) = ln
            # if abs(tx) >= 1 or abs(ty) >= 1:
            #    return False
            if lx == 0:  # vertical
                return 0 <= px < 1
            y0 = -px * ly / lx + py
            y1 = (1 - px) * ly / lx + py
            return (y0 < 1 or y1 < 1) and (y0 >= 0 or y1 >= 0)
        # Translation, Identity
        return True

    @functools.cache
    def symm_group(self) -> Tuple[str, Optional[int]]:
        eig = self.mat.eigenvects()
        pts = [v[:2] for λ, _, vs in eig if λ == 1 for v in vs if v[2] != 0]
        vec = [v[:2] for λ, _, vs in eig if λ == 1 for v in vs if v[2] == 0]
        if len(vec) == 2:
            return ("𝜏", None)
        elif len(vec) == 1:
            if len(pts) == 1:
                return ("𝜎", 2)
            else:
                return ("𝛾", None)
        elif len(vec) == 0:
            if len(pts) == 1:
                for order in (2, 3, 4, 6):
                    if self.mat ** order == sp.eye(3):
                        break
                else:
                    raise ValueError("None crystallographic rotation")
                return ("𝜌", order)
        print(eig)
        raise ValueError("Bug! Impossible point group")

    def description(self):
        sg = self.symm_group()[0]
        if sg == "𝜏":
            return f"translation by {tuple(self.mat[:2,2])}"
        if sg == "𝜌":
            pt = self.fixed_point()
            for k in (2, 3, 4, 6):
                if self.mat ** k == sp.eye(3):
                    break
            return f"{k}-fold rotation around {tuple(pt)}"
        p, v, t = self.stable_line()
        if sg == "𝜎":
            return f"reflection across {format_line(p,v)}"
        return f"glide reflection across {format_line(p,v)} by {tuple(t)}"

    def inv(self):
        if self.name is None:
            name = None
        elif "*" in self.name or " " in self.name:
            name = f"({self.name})⁻¹"
        else:
            name = f"{self.name}⁻¹"
        return MatrixOperator(self.mat.inv(), name=name)


def format_line(point, direction):
    "convert point/vector representation of a line into an equation like 'x + y = 0'"
    x0, y0 = point
    lx, ly = direction
    x, y = sp.var("x y")
    return sp.printing.pretty(sp.Eq(ly * x - lx * y, ly * x0 - lx * y0).simplify())


def complete_group(generators):
    "Given a list of generators in fractional coordinates, complete remaining operators"
    ops = dict()  # Use dict instead of set to preserve insertion order
    # Note: omits translational generators from results
    for g in generators:
        if g.in_unit_cell():
            ops[g] = None
        i = g.inv()
        if i.in_unit_cell():
            ops[i] = None
    n = 0
    while n < len(ops):
        n = len(ops)
        if n > 2 ** 8:
            raise Exception("Group too large")

        for op in list(ops.keys()):
            for gen in itertools.chain(
                generators, list(ops.keys())
            ):  # itertools.chain(generators, (g.inv() for g in generators)):
                new = gen * op

                # print(f"Considering {new}")
                if new.in_unit_cell() and new not in ops:
                    # print(f"Adding {new}")
                    ops[gen * op] = None
                # elif new in ops:
                #    print(f"Skipping {new} equivalent to {[g for g in ops.keys() if g == new][0]}")
        # print(f"Current ops: {{{', '.jo|in(map(str,ops.keys()))}}}")

    return list(ops.keys())
