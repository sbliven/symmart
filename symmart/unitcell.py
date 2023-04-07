from typing import Dict, List, Optional, Tuple

import numpy as np
import svgwrite
import sympy as sp

from .operators import MatrixOperator, complete_group


class CellDiagram:
    def __init__(self, a: complex, b: complex, **kwargs):
        self.a = a
        self.b = b
        self.margin = 10
        self._diagram = svgwrite.Drawing(**kwargs)
        self._coord = self._diagram.add(
            self._diagram.g(transform=f"matrix(1 0 0 -1 0 0)")
        )

    def draw_cell(self):
        corners = np.array((0, self.a, self.a + self.b, self.b))
        xmin, xmax = corners.real.min(), corners.real.max()
        ymin, ymax = corners.imag.min(), corners.imag.max()
        # self._coord.add(
        #     self._diagram.rect(
        #         (xmin - self.margin, ymin - self.margin),
        #         (xmax - xmin + self.margin * 2, ymax - ymin + self.margin * 2),
        #         fill="yellow",
        #     )
        # )

        self._diagram.viewbox(
            xmin - self.margin,
            ymin - ymax - self.margin,
            xmax - xmin + self.margin * 2,
            ymax - ymin + self.margin * 2,
        )
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            self._coord.add(
                self._diagram.line(
                    start=(start.real, start.imag),
                    end=(end.real, end.imag),
                    stroke_width="1",
                    stroke="black",
                )
            )
        self._coord.add(self._diagram.circle(center=(0, 0), r=1, fill="red"))
        return self._diagram

    def draw_ops(self, ops, expand=True):
        if expand:
            ops = complete_group(ops)

        def sg_order(op):
            return "𝜏𝜎𝛾𝜌".index(op.symm_group()[0])

        for op in sorted(nonredundant(ops), key=sg_order):
            sg, order = op.symm_group()
            if sg == "𝜏":
                pass
            elif sg == "𝜌":
                if order == 2:
                    self._draw_rot2(op)
                else:
                    self._draw_rot(op, order)
            elif sg == "𝜎":
                self._draw_reflection(op)
            elif sg == "𝛾":
                self._draw_glide(op)
            else:
                raise ValueError(f"Unimplemented operator {sg}{order}")
        return self._diagram

    def _draw_rot2(self, op):
        fixed_lat = op.fixed_point()  # lattice coords
        fixed_real = self.a * float(fixed_lat[0]) + self.b * float(fixed_lat[1])
        size = 5
        self._coord.add(
            self._diagram.ellipse(
                center=(fixed_real.real, fixed_real.imag), r=(size / 2, size)
            )
        )
        return self._diagram

    def _draw_rot(self, op, n):
        fixed_lat = op.fixed_point()  # lattice coords
        fixed_real = self.a * float(fixed_lat[0]) + self.b * float(fixed_lat[1])
        r = 5
        points = [
            fixed_real
            + r * np.exp(2j * np.pi * (i / n + 1 / 4 + ((n + 1) % 2) / 2 / n))
            for i in range(n)
        ]
        self._coord.add(self._diagram.polygon([(p.real, p.imag) for p in points]))
        return self._diagram

    def _draw_reflection(self, op):
        spacing = 1  # stroke width

        (px, py), (lx, ly), _ = op.stable_line()
        t = line_in_bb(
            px,
            py,
            lx,
            ly,
            -self.margin / abs(self.a),
            1 + self.margin / abs(self.a),
            -self.margin / abs(self.b),
            1 + self.margin / abs(self.b),
        )
        if t is None:
            return self._diagram
        t0, t1 = t

        start = self.a * float(px + lx * t0) + self.b * float(py + ly * t0)
        end = self.a * float(px + lx * t1) + self.b * float(py + ly * t1)

        self._coord.add(
            self._diagram.line(
                start=(start.real, start.imag),
                end=(end.real, end.imag),
                stroke_width=spacing * 3,
                stroke="white",
            )
        )

        norm = spacing * 1j * (end - start) / abs(end - start)

        self._coord.add(
            self._diagram.line(
                start=((start + norm).real, (start + norm).imag),
                end=((end + norm).real, (end + norm).imag),
                stroke_width=spacing,
                stroke="grey",
            )
        )
        self._coord.add(
            self._diagram.line(
                start=((start - norm).real, (start - norm).imag),
                end=((end - norm).real, (end - norm).imag),
                stroke_width=spacing,
                stroke="grey",
            )
        )

        return self._diagram

    def _draw_glide(self, op):
        (px, py), (lx, ly), _ = op.stable_line()
        t = line_in_bb(
            px,
            py,
            lx,
            ly,
            -self.margin / abs(self.a),
            1 + self.margin / abs(self.a),
            -self.margin / abs(self.b),
            1 + self.margin / abs(self.b),
        )
        if t is None:
            return self._diagram
        t0, t1 = t

        start = self.a * float(px + lx * t0) + self.b * float(py + ly * t0)
        end = self.a * float(px + lx * t1) + self.b * float(py + ly * t1)

        self._coord.add(
            self._diagram.line(
                start=(start.real, start.imag),
                end=(end.real, end.imag),
                stroke_width="1",
                stroke="grey",
                stroke_dasharray="4,4",
            )
        )
        return self._diagram


def nonredundant(ops: List[MatrixOperator]):
    by_sg: Dict[str, List[Tuple[MatrixOperator, Optional[int]]]] = {}
    # Split by symm group
    for op in ops:
        sg, order = op.symm_group()
        by_sg.setdefault(sg, []).append((op, order))
    if "𝜌" in by_sg:
        by_fp: Dict[
            Tuple[sp.core.Number, sp.core.Number], Tuple[MatrixOperator, Optional[int]]
        ] = {}
        for op, order in by_sg["𝜌"]:
            fp = tuple(op.fixed_point())
            if fp not in by_fp or by_fp[fp][1] < order:
                by_fp[fp] = (op, order)
        # import pdb;pdb.set_trace()

        by_sg["𝜌"] = by_fp.values()
    # print(by_sg)
    return [op for sg in by_sg.values() for op, _ in sg]


def line_in_bb(px, py, lx, ly, bb_x0=0, bb_x1=1, bb_y0=0, bb_y1=1):
    """Intersect a line with a bounding box

    The bounding box is closed.

    Returns (t0,t1) such that (px,py)+(lx,ly)*t is inside the bounding box,
    or None if they don't intersect
    """
    # vertical
    if lx == 0:
        if ly == 0:
            raise ValueError("Not a valid line")
        if bb_x0 <= px <= bb_x1:
            return (bb_y0 - py) / ly, (bb_y1 - py) / ly
        else:
            return None
    # horizontal
    if ly == 0:
        if bb_y0 <= py <= bb_y1:
            return (bb_x0 - px) / lx, (bb_x1 - px) / lx
        else:
            return None

    # diagonal
    # x = (bb_x0, bb_x1, lx / ly * (bb_y0 - py) + px, lx / ly * (bb_y1 - py) + px)
    # y = (ly / lx * (bb_x0 - px) + py, ly / lx * (bb_x1 - px) + py, bb_y0, bb_y1)
    # t = ((bb_x0 - px) / lx, (bb_x1 - px) / lx, (bb_y0 - py) / ly, (bb_y1 - py) / ly)

    valid_t = set()
    if bb_y0 <= ly / lx * (bb_x0 - px) + py <= bb_y1:  # left
        valid_t.add((bb_x0 - px) / lx)
    if bb_y0 <= ly / lx * (bb_x1 - px) + py <= bb_y1:  # right
        valid_t.add((bb_x1 - px) / lx)
    if bb_x0 <= lx / ly * (bb_y0 - py) + px <= bb_x1:  # bottom
        valid_t.add((bb_y0 - py) / ly)
    if bb_x0 <= lx / ly * (bb_y1 - py) + px <= bb_x1:  # top
        valid_t.add((bb_y1 - py) / ly)
    if len(valid_t) != 2:  # Either non-intersecting or clips a corner
        return None
    return tuple(sorted(valid_t))