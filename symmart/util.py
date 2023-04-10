import math
from typing import Tuple

import numpy as np

unit_box = np.array([-1 - 1j, 1 + 1j])


def grid_points(height, width, limits=unit_box, fix_aspect=True):
    """x,y grid points for use building a 2D grid of complex numbers

    Args:
    - height, width: image dimensions
    - limits: tuple of complex numbers giving bottom-left and upper-right corners of the
      image
    - fix_aspect: if True, expands the larger axis to preserve a 1:1 aspect ratio

    Returns
    Arrays (x,y) representing linearly spaced coordinates along each dimension.
    """
    xmin, xmax = limits[0].real, limits[1].real
    ymin, ymax = limits[0].imag, limits[1].imag
    w = width - 1
    h = height - 1
    if fix_aspect:
        scalex = (xmax - xmin) / w  # unit/px
        scaley = (ymax - ymin) / h
        scale = min(scalex, scaley)

        xmid = xmin + xmax
        ymid = ymin + ymax

        xmin = (xmid - scale * w) / 2
        xmax = (xmid + scale * w) / 2

        ymin = (ymid - scale * h) / 2
        ymax = (ymid + scale * h) / 2

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymax, ymin, height)
    return x, y


def complex_grid(height, width, limits=unit_box, fix_aspect=True):
    """Create a 2D grid of complex numbers

    This inverts the y axis, so complex numbers increase from [0,h] towards [w,0].

    Pixels are evaluated at their center. The limits are also given at pixel centers.

    Args:
    - height, width: image dimensions
    - limits: tuple of complex numbers giving bottom-left and upper-right corners of the
      image
    - fix_aspect: if True, expands the larger axis to preserve a 1:1 aspect ratio
    """
    x, y = grid_points(height, width, limits=limits, fix_aspect=fix_aspect)
    zx, zy = np.meshgrid(x, y, sparse=True)
    z = zx + zy * 1j
    return z


def loground(x: float, base=1000, sigfigs=1) -> Tuple[int, int]:
    """Rounds x on a log scale to the nearest power.

    For `mantissa, exponent = loground(x, base)`,

        x ≈ mantissa * base ** exponent

    Args:
        x: input number.
        base: Base of the log scale. Defaults to 1000 to give thousands, millions,
            billions, etc.
        sigfigs: Number of significant figures

    Returns:
    Tuple with the mantissa and exponent

    """
    if sigfigs < 1:
        raise ValueError("Require strictly positive sigfigs")

    if abs(x) <= base ** -8:  # effective 0
        return (0, 0)

    # Choose log function for numeric stability
    if base % 10 == 0:  # probably power of 10

        def logB(xx):
            return math.log10(xx) / math.log10(base)

    else:  # probably power of 2

        def logB(xx):
            return math.log2(xx) / math.log2(base)

    exponent = math.floor(logB(abs(x)))  # chosen so mantissa is in [1, base]
    mantissa = round(x * base ** (sigfigs - 1 - exponent)) * base ** (1 - sigfigs)

    # For negative numbers, floor and round might be opposite directions
    if abs(mantissa) >= base:
        # Example: autoUnit_loground(-99.9, base=10, sigfigs=2)
        mantissa //= base
        exponent += 1

    assert 1.0 <= abs(mantissa) < base, mantissa

    return (mantissa, exponent)


def metricunit(x: int, base=1000, sigfigs=1) -> str:
    """Converts x to a compact string using metric units

    Metric units are used irrespective of the base (e.g. 'k' for the first power of
    base)

    Args:
        x: input number.
        base: Base of the log scale. Defaults to 1000 to give thousands, millions,
            billions, etc.
        sigfigs: Number of significant figures

    Returns:
    A string with the rounded number followed by a metric prefix (k, M, G, T, etc).

    Raise: IndexOutOfBounds if `abs(x)` is outside the range of supported prefixes.
    """
    prefixes = " kMGTPEZYyzafpµmcd"  # wraps around, so need to check bounds

    mantissa, exponent = loground(x, base, sigfigs)

    if abs(exponent) > 8:
        raise ValueError(
            "Rounded number outside range. No prefix for {}**{}".format(base, exponent)
        )

    prefix = prefixes[exponent] if exponent != 0 else ""
    return "{:.{}f}{}".format(mantissa, sigfigs - 1, prefix)
