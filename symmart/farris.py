import functools

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .util import unit_box


def grid_points(height, width, limits=unit_box, fix_aspect=True):
    """x,y grid points for use building a 2D grid of complex numbers"""
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


def image_from_wheel(height, width, *ops, wheel, limits=unit_box, fix_aspect=True):
    """Create an image from a color wheel function

    Args:
    - height, width: image dimensions
    - ops: functions ℭ→ℭ to apply to the coordinates
    - wheel: vectorized function mapping complex point to (r,g,b) tuples. Outputs an Nx3
      array
    - origin: location of the origin within the image, as a fraction of the image, with
      (0,0) meaning bottom left
    - scale: width of the smaller dimension, in complex units
    """
    z = complex_grid(height, width, limits=limits, fix_aspect=fix_aspect)
    z = functools.reduce(lambda z, op: op(z), ops, z)
    v = wheel(z.flatten())
    return v.reshape(*z.shape, 3)


def show_plane_fn(
    *ops,
    wheel,
    limits=unit_box,
    height=300,
    width=300,
    fix_aspect=True,
    outfile=None,
    **kwargs
):
    img = image_from_wheel(
        height, width, *ops, wheel=wheel, limits=limits, fix_aspect=False
    )
    if outfile:
        Image.fromarray(img).save(outfile)
    return plt.imshow(
        img,
        extent=(limits[0].real, limits[1].real, limits[0].imag, limits[1].imag),
        aspect="equal" if fix_aspect else "auto",
        **kwargs
    )
