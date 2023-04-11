"""Functions for creating color wheels.

A color wheel is a function that maps a complex number to a color, often a gradient or
picture.
"""

import numpy as np
from importlib_resources import files  # backport for python < 3.9
from PIL import Image
from scipy import interpolate

from .util import grid_points, unit_box

# Synthetic wheels


def wheel_6(z, radius=1.0):
    """Color wheel with 6-fold hues, white near 0, and fading to black beyond radius"""
    rad = np.abs(z)
    sq3 = np.sqrt(3)
    hue = (np.clip(1 - 2 * (rad / radius - 1), 0, 1) * 255).astype(np.uint8)
    radial = (np.clip(1 - (rad / radius) ** 2, 0, 1) * 255).astype(np.uint8)
    r = np.where(z.real >= 0, hue, radial)
    g = np.where(z.imag * sq3 >= z.real, hue, radial)
    b = np.where(z.imag * sq3 <= -z.real, hue, radial)
    return np.stack((r, g, b), axis=1)


def wheel_gradiant(z, radius=1.0, hue_steps=6, white_radius=0.2, hue_round=np.round):
    """Color wheel with n-fold hues, white near 0, and fading to black beyond radius

    This mimics the wheel in Figure 6.1b of Farris.

    Colors start with red centered along the positive real axis with hue increasing
    counterclockwise. After a sharp white circle at the center, saturated color bands
    fade to black at 2*radius.

    Args:
    - radius: half the radius of full-black
    - hue_steps: number of colors
    """
    r = np.abs(z)
    θ = np.arctan2(z.imag, z.real) / 2 / np.pi  # in turns
    h = hue_round(θ * hue_steps) / hue_steps
    s = 1
    slope = 0.25 / (white_radius - radius)
    intercept = 0.5 - slope * white_radius
    l = np.where(r < white_radius, 1, np.maximum(slope * r + intercept, 0))
    return hsl2rgb(h, s, l)


def wheel_stepped(
    z, radius=1.0, hue_steps=6, l_steps=6, hue_round=np.round, l_round=np.round
):
    """Color wheel with n-fold hues, white near 0, and fading to black beyond radius

    Colors start with red centered along the positive real axis with hue increasing
    counterclockwise. Lightness decreases linearly from white at the center to black
    at 2*radius, leaving fully saturated colors at 1*radius. This is descritized to
    rsteps-levels using a rounding funtion (so that the central white disk has the
    same diameter as the color bands). Use odd rsteps to give a discontinuity at
    radius.

    Args:
    - radius: radius for fully saturated hues (or half the radius of full-black)
    - steps: number of colors
    - rsteps: number of hues
    """
    r = np.abs(z)
    θ = np.arctan2(z.imag, z.real) / 2 / np.pi  # in turns
    h = hue_round(θ * hue_steps) / hue_steps
    s = 1
    l = l_round((1 - np.minimum(r / radius / 2, 1)) * l_steps) / l_steps

    return hsl2rgb(h, s, l)


def axes_wheel(z, thickness=0.2, length=1):
    r = np.where(
        (np.abs(z.imag) <= thickness / 2) & (0 <= z.real) & (z.real <= length), 255, 0
    )
    g = np.where(
        (np.abs(z.real) <= thickness / 2) & (0 <= z.imag) & (z.imag <= length), 255, 0
    )
    b = np.zeros_like(z)
    return np.stack((r, g, b), axis=1).astype(np.uint8)


def hsl_wheel(z, radius=1.0):
    r = np.abs(z)
    θ = np.arctan2(z.imag, z.real)
    h = θ / 2 / np.pi
    s = 1
    l = 1 - np.minimum(r / radius / 2, 1)
    return hsl2rgb(h, s, l)


def hsl2rgb(h, s, l):
    "Convert HSL (0 to 1) to RGB (0 to 255)"
    h = np.atleast_1d(h) % 1
    a = s * np.minimum(l, 1 - l)
    rgb = np.zeros((*np.broadcast(h, s, l).shape, 3))
    # import pdb;pdb.set_trace()
    for i, n in enumerate((0, 8, 4)):
        k = (n + h * 12) % 12
        f = l - a * np.maximum(-1, np.minimum(np.minimum(k - 3, 9 - k), 1))
        rgb[:, i] = f
    return rgb


# Image wheel


def image_wheel(filename: str, limits=unit_box, background=(0, 0, 0), fix_aspect=True):
    """A color wheel derived from an image

    Args:
    - filename: image filename
    - limits: 2-tuple giving the range of the wheel in complex space
    """
    img = np.asarray(Image.open(filename).convert("RGB"))
    return raster_wheel(
        img, limits=limits, background=background, fix_aspect=fix_aspect
    )


# def raster_wheel(img, limits=(-1 - 1j, 1 + 1j),kind="linear"):
#     x, y = grid_points(img.shape[0], img.shape[1], limits=limits, fix_aspect=True)
#     r = interpolate.interp2d(x, y, img[:,:,0], kind=kind)
#     g = interpolate.interp2d(x, y, img[:,:,1], kind=kind)
#     b = interpolate.interp2d(x, y, img[:,:,2], kind=kind)

#     def wheel(r,g,b,z):
#         return np.stack([np.concatenate([f(x,y) for f in (r,g,b)]).astype(np.uint8)
#              for x,y in zip(z.real, z.imag)])

#     return functools.partial(wheel, r,g,b)


def raster_wheel(img, limits=unit_box, background=(0, 0, 0), fix_aspect=True):
    """A color wheel derived from a MxNx3 matrix

    Args:
    - img: MxNx3 matrix following image conventions (origin in the top left)
    - limits: bounding box to contain the image
    """
    h, w, _ = img.shape
    x, y = grid_points(h, w, limits=limits, fix_aspect=fix_aspect)
    y = np.flip(y)
    img = img.transpose(1, 0, 2)
    img = np.flip(img, axis=1)
    kind = "linear"
    r = interpolate.RegularGridInterpolator(
        (x, y), img[:, :, 0], method=kind, bounds_error=False, fill_value=background[0]
    )
    g = interpolate.RegularGridInterpolator(
        (x, y), img[:, :, 1], method=kind, bounds_error=False, fill_value=background[1]
    )
    b = interpolate.RegularGridInterpolator(
        (x, y), img[:, :, 2], method=kind, bounds_error=False, fill_value=background[2]
    )

    def wheel(z):
        pts = np.stack((z.real, z.imag), axis=-1)
        return np.stack([f(pts) for f in (r, g, b)], axis=1).astype(np.uint8)

    return wheel


# Wheel transformations


def invert_wheel(wheel):
    """Modify a color wheel to invert all colors"""
    return lambda z: 255 - wheel(z)


def reverse_wheel(wheel, strip_width=0.05, stripe_color=(0, 0, 0)):
    """Modify a color wheel so that -z is the inverse of z.

    Note that the part of the original wheel below the real axis will be discarded.

    A strip of (usually neutral) color can be placed along the axis.
    """

    def modify(z):
        neg = z.imag < 0
        colors = wheel(np.where(neg, -z, z))
        colors[neg, :] = 255 - colors[neg, :]
        colors[
            (-strip_width / 2 < z.imag) & (z.imag < strip_width / 2), :
        ] = np.asarray(stripe_color).reshape((1, 3))
        return colors

    return modify


def reflect_x_wheel(wheel, strip_width=0.05, stripe_color=(0, 0, 0)):
    """Modify a color wheel so that z̄ is the inverse of z.

    Note that the part of the original wheel below the real axis will be discarded.

    A strip of (usually neutral) color can be placed along the axis.
    """

    def modify(z):
        neg = z.imag < 0
        colors = wheel(z.real + np.abs(z.imag) * 1j)
        colors[neg, :] = 255 - colors[neg, :]
        colors[
            (-strip_width / 2 < z.imag) & (z.imag < strip_width / 2), :
        ] = np.asarray(stripe_color).reshape((1, 3))
        return colors

    return modify


class LazyWheel:
    """Lazily loads an image wheel from the package data files"""

    def __init__(self, path, **kwargs):
        """
        Args:
        - path: name of an image file in symmart.images
        - kwargs: args to pass to image_wheel
        """
        self._wheel = None
        self._path = path
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if self._wheel is None:
            self._wheel = image_wheel(
                files("symmart.images").joinpath(self._path), **self._kwargs
            )
        return self._wheel(*args, **kwargs)
