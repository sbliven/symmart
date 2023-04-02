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
