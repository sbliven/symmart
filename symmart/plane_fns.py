import functools

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .util import unit_box, complex_grid


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


def torus(limits=unit_box):
    """Wrap the complex plane to a rectangular patch"""
    period = limits[1] - limits[0]
    origin = limits[0]

    def wheel(z):
        return (
            np.mod(z.real - origin.real, period.real)
            + 1j * np.mod(z.imag - origin.imag, period.imag)
            + origin
        )

    return wheel


def translate(delta):
    return lambda z: z + delta
