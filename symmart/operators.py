"""Manipulate symmetry groups algebraically
"""
import functools
import itertools
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp


class MatrixOperator:
    """Matrix-based implementation of an abstract group operator"""

    def __init__(self, mat, name=None):
        if isinstance(mat, MatrixOperator):
            self.mat = mat.mat
        else:
            self.mat = sp.Matrix(mat)
        self.name = name

    def __mul__(self, other):
        "Alias for matrix multiplication (self @ other)"
        return self @ other

    def __matmul__(self, other):
        """Operator composition: (f@g)(z) = f(g(z))

        For other uses, delegates to self.mat @ other
        """
        if isinstance(other, MatrixOperator):
            if self.name and other.name:
                name = f"{self.name} * {other.name}"
            else:
                name = None
            return MatrixOperator(self.mat * other.mat, name=name)
        return self.mat @ other

    def __pow__(self, pow):
        """Matrix power"""
        if self.name:
            name = f"{self.name} ** {pow}"
        else:
            name = None
        return MatrixOperator(self.mat ** pow, name=name)

    def __eq__(self, other):
        if isinstance(other, MatrixOperator):
            return self.mat == other.mat
        return False

    def __hash__(self):
        return hash(tuple(self.mat.flat()))

    def __call__(self, z):
        return self.mat * np.asarray(z)

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
        """Check if the stable point/line of this operation is inside the unit cell

        Defined as:
        - point groups: the fixed point is in the unit cell.
          Note that powers of a point group are also in the unit cell (eg both 𝜌4 and
          𝜌2).
        - reflection: Special cases were chosen (arbitrarily) for each stable line
            - Horizontal: y = {0,1,2,3}/4
            - Vertical: x = {0,1,2,3}/4
            - Origin: x+y=0, x-2y=0, x-y=0, or 2x-y=0
            - Diagonal: x+y=1/2, x-y=1/2, 2x-y=1/2, x-2y=1/2
        - glide: same stable line constraints as reflections, plus
            - One translational component must be (0,1/2), (1/2, 0), (1/2, ±1/2),
              (1, 1/2) or (1/2, 1)
        - translation: 1 or i only
        - identity: False

        """
        # Point symmetry
        pt = self.fixed_point()
        if pt is not None:
            return 0 <= pt[0] < 1 and 0 <= pt[1] < 1
        # reflection/glide
        ln = self.stable_line()
        if ln is not None:
            (px, py), (lx, ly), (tx, ty) = ln
            # Point on line closest to the origin
            # Project p onto a vector perpendicular to l
            q = (lx * py - ly * px) / (lx * lx + ly * ly)
            qx, qy = -ly * q, lx * q

            # Glide must be one half in some direction and at most 1 in the other
            half = sp.sympify(1) / 2
            # if not ((0 <= tx <= half) and (-half < ty <= half)):
            #     return False
            if tx == half:
                if ty not in (-half, 0, half, 1):
                    return False
            elif ty == half:
                if tx not in (0, half, 1):
                    return False
            elif tx != 0 or ty != 0:
                return False

            # This should cover all possible orientations
            quarter = sp.sympify(1) / 4
            if qx == 0 and 0 <= qy < 1:  # horizontal (or through origin)
                return True
            if qy == 0 and 0 <= qx < 1:  # vertical
                return True
            # diagonals: x+y=1/2, x-y=1/2, x-2y=1/2, 2x-y=1/2
            if 0 < qx <= quarter and -quarter <= qy <= quarter:
                return True
            return False

            #
            # return (0 <= qx < 1) and (0 <= qy < 1) and

            # if abs(tx) >= 1 or abs(ty) >= 1:
            #    return False
            # if lx == 0:  # vertical
            #     return 0 <= px < 1
            # y0 = -px * ly / lx + py
            # y1 = (1 - px) * ly / lx + py
            # return (y0 < 1 or y1 < 1) and (y0 >= 0 or y1 >= 0)
        # Translation, Identity
        if self.mat[0, 2] == 0:
            return self.mat[1, 2] == 1
        elif self.mat[0, 2] == 1:
            return self.mat[1, 2] == 0
        return False

    @functools.cache
    def symm_group(self) -> Tuple[str, Optional[int]]:
        """Get the point group of this operator (as a greek letter)

        Returns:
        - 𝜏, translation (or identity)
        - 𝜌, rotation
        - 𝜎, reflection
        - 𝛾, glide reflection
        """
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
                    raise ValueError("Non-crystallographic rotation")
                return ("𝜌", order)
        raise ValueError("Bug! Impossible point group")  # pragma: no cover

    def description(self):
        "English description of this operator"
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

    def inv(self) -> "MatrixOperator":
        "Get inverse"
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
        if n > 2 ** 6:  # Indicates a bug, such as noncrystallographic generators
            raise Exception("Group too large")  # pragma: no cover

        for op in list(ops.keys()):
            for gen in itertools.chain(
                generators, list(ops.keys())
            ):  # itertools.chain(generators, (g.inv() for g in generators)):
                new = gen * op

                # print(f"Considering {new}")
                if new not in ops and new.in_unit_cell():
                    # print(f"Adding {new}")
                    ops[gen * op] = None
                # elif new in ops:
                #     print(
                #         f"Skipping {new} equivalent to "
                #         f"{[g for g in ops.keys() if g == new][0]}"
                #     )
        # print(f"Current ops: {{{', '.join(map(str,ops.keys()))}}}")

    return list(ops.keys())
