from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import dash_svg as dsvg
import numpy as np
import svgwrite
import sympy as sp

from .operators import MatrixOperator, complete_group


class CellDiagram(ABC):
    def __init__(self, a: complex, b: complex, **kwargs):
        self.a = a
        self.b = b
        self.margin = 10

    @abstractmethod
    def _set_viewbox(self, minx=0, miny=0, width=0, height=0):
        ...

    @abstractmethod
    def _draw_ellipse(self, center, r):
        ...

    @abstractmethod
    def _draw_line(
        self, start, end, stroke_width=None, stroke=None, stroke_dasharray=None
    ):
        ...

    @abstractmethod
    def _draw_circle(self, center, r, fill=None):
        ...

    @abstractmethod
    def _draw_polygon(self, points):
        ...

    @abstractmethod
    def _draw_rect(self, corner1, size, fill=None):
        ...

    def draw_cell(self):
        corners = np.array((0, self.a, self.a + self.b, self.b))
        xmin, xmax = corners.real.min(), corners.real.max()
        ymin, ymax = corners.imag.min(), corners.imag.max()
        w = xmax - xmin
        h = ymax - ymin

        # Background
        self._draw_rect(
            (xmin - self.margin, ymin - self.margin),
            (w + self.margin * 2, h + self.margin * 2),
            fill="white",
        )

        self._set_viewbox(
            xmin - self.margin,
            -ymin - h - self.margin,
            w + self.margin * 2,
            h + self.margin * 2,
        )
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            self._draw_line(
                start=(start.real, start.imag),
                end=(end.real, end.imag),
                stroke_width="1",
                stroke="black",
            )
        self._draw_circle(center=(0, 0), r=1, fill="red")
        return self

    def draw_ops(self, ops, expand=True):
        "Draw unit cell operators"
        if expand:
            ops = complete_group(ops)
            ops = expand_group(ops)

        def sg_order(op):
            "Layer order for ops"
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
        return self

    @abstractmethod
    def draw(self):
        ...

    def _draw_rot2(self, op):
        fixed_lat = op.fixed_point()  # lattice coords
        fixed_real = self.a * float(fixed_lat[0]) + self.b * float(fixed_lat[1])
        size = 5
        self._draw_ellipse(
            center=(fixed_real.real, fixed_real.imag), r=(size / 2, size)
        )
        return self

    def _draw_rot(self, op, n):
        fixed_lat = op.fixed_point()  # lattice coords
        fixed_real = self.a * float(fixed_lat[0]) + self.b * float(fixed_lat[1])
        r = 5
        points = [
            fixed_real
            + r * np.exp(2j * np.pi * (i / n + 1 / 4 + ((n + 1) % 2) / 2 / n))
            for i in range(n)
        ]
        self._draw_polygon([(p.real, p.imag) for p in points])
        return self

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

        self._draw_line(
            start=(start.real, start.imag),
            end=(end.real, end.imag),
            stroke_width=spacing * 3,
            stroke="white",
        )

        norm = spacing * 1j * (end - start) / abs(end - start)

        self._draw_line(
            start=((start + norm).real, (start + norm).imag),
            end=((end + norm).real, (end + norm).imag),
            stroke_width=spacing,
            stroke="grey",
        )

        self._draw_line(
            start=((start - norm).real, (start - norm).imag),
            end=((end - norm).real, (end - norm).imag),
            stroke_width=spacing,
            stroke="grey",
        )

        return self

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

        self._draw_line(
            start=(start.real, start.imag),
            end=(end.real, end.imag),
            stroke_width="1",
            stroke="grey",
            stroke_dasharray="4,4",
        )

        return self


class SvgwriteCellDiagram(CellDiagram):
    def __init__(self, a: complex, b: complex, **kwargs):
        super().__init__(a, b)
        self._diagram = svgwrite.Drawing(**kwargs)
        self._coord = self._diagram.add(
            self._diagram.g(transform="matrix(1 0 0 -1 0 0)")
        )

    def tostring(self):
        return self._diagram.tostring()

    def draw(self):
        return self._diagram

    def _set_viewbox(self, minx=0, miny=0, width=0, height=0):
        self._diagram.viewbox(minx, miny, width, height)

    def _draw_ellipse(self, center, r):
        self._coord.add(self._diagram.ellipse(center=center, r=r))

    def _draw_line(
        self, start, end, stroke_width=None, stroke=None, stroke_dasharray=None
    ):
        args = dict(
            stroke_width=stroke_width, stroke=stroke, stroke_dasharray=stroke_dasharray
        )
        self._coord.add(
            self._diagram.line(
                start=start, end=end, **{k: v for k, v in args.items() if v is not None}
            )
        )

    def _draw_circle(self, center, r, fill=None):
        args = {k: v for k, v in [("fill", fill)] if v is not None}
        self._coord.add(self._diagram.circle(center=center, r=r, **args))

    def _draw_polygon(self, points):
        self._coord.add(self._diagram.polygon(points))

    def _draw_rect(self, corner1, size, fill=None):
        args = {k: v for k, v in [("fill", fill)] if v is not None}
        self._coord.add(self._diagram.rect(corner1, size, **args))


class DashCellDiagram(CellDiagram):
    def __init__(self, a: complex, b: complex, **kwargs):
        super().__init__(a, b, **kwargs)
        self._coord = dsvg.G([], transform="matrix(1 0 0 -1 0 0)")
        self._diagram = dsvg.Svg([self._coord], viewBox="0 0 100 100", **kwargs)

    def draw(self):
        return self._diagram

    def _set_viewbox(self, minx=0, miny=0, width=0, height=0):
        self._diagram.viewBox = f"{minx} {miny} {width} {height}"

    def _draw_ellipse(self, center, r):
        self._coord.children.append(
            dsvg.Ellipse(cx=center[0], cy=center[1], rx=r[0], ry=r[1])
        )

    def _draw_line(
        self, start, end, stroke_width=None, stroke=None, stroke_dasharray=None
    ):
        args = dict(
            strokeWidth=stroke_width, stroke=stroke, strokeDasharray=stroke_dasharray
        )
        self._coord.children.append(
            dsvg.Line(
                x1=start[0],
                y1=start[1],
                x2=end[0],
                y2=end[1],
                **{k: v for k, v in args.items() if v is not None},
            )
        )

    def _draw_circle(self, center, r, fill="red"):
        self._coord.children.append(
            dsvg.Circle(cx=center[0], cy=center[1], r=str(r), fill=fill)
        )

    def _draw_polygon(self, points):
        pts = " ".join(f"{x},{y}" for x, y in points)
        self._coord.children.append(dsvg.Polygon(points=pts))

    def _draw_rect(self, corner1, size, fill=None):
        args = {k: v for k, v in [("fill", fill)] if v is not None}
        x1, y1 = corner1
        w, h = size
        self._coord.children.append(dsvg.Rect(x=x1, y=y1, width=w, height=h, **args))


def nonredundant(ops: List[MatrixOperator]):
    "Remove overlapping rotation operators"
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


def expand_group(ops: List[MatrixOperator]):
    𝜏1 = MatrixOperator([[1, 0, 1], [0, 1, 0], [0, 0, 1]], "𝜏1")
    𝜏i = MatrixOperator([[1, 0, 0], [0, 1, 1], [0, 0, 1]], "𝜏i")

    for op in ops:
        yield op
        fp = op.fixed_point()
        if fp is not None:
            if fp[0] == 0:
                yield 𝜏1 * op * 𝜏1.inv()
            if fp[1] == 0:
                yield 𝜏i * op * 𝜏i.inv()
            if fp == [0, 0]:
                yield 𝜏1 * 𝜏i * op * 𝜏i.inv() * 𝜏1.inv()
        else:
            ln = op.stable_line()
            if ln is not None:
                (px, py), (lx, ly), (tx, ty) = ln
                # normal = (-ly, lx)
                # Choose 𝜏1 or 𝜏i based on projection to normal
                if lx == 0 or (0 < abs(ly) <= abs(lx)):
                    𝜏 = 𝜏1
                    𝜏_proj = -ly
                else:
                    𝜏 = 𝜏i
                    𝜏_proj = lx
                p_proj = py * lx - px * ly  # project p onto normal

                # Unit box corners, projected onto the normal vector
                bounds = [0, -ly, lx, lx - ly]
                b1 = (min(bounds) - p_proj) / 𝜏_proj
                b2 = (max(bounds) - p_proj) / 𝜏_proj
                if b1 > b2:
                    b1, b2 = b2, b1
                # min(bounds) ≤ i*𝜏_proj + p_proj ≤ max(bounds)
                for i in range(int(np.ceil(b1)), int(np.floor(b2)) + 1):
                    if i != 0:
                        yield 𝜏 ** i * op * 𝜏 ** -i


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
