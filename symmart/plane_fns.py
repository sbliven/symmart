import base64
import functools
from io import BytesIO
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from .util import complex_grid, unit_box


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
    **kwargs,
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
        **kwargs,
    )


def plane_fn_bytes(
    *ops,
    wheel,
    limits=unit_box,
    height=300,
    width=300,
    fix_aspect=True,
    outfile=None,
    format="png",
    **kwargs,
) -> bytes:
    """Convert a plane function to a formated bytes object

    This can be quickly displayed using ipywidgets.Image(value=plane_fn_bytes()),
    which is generally faster than the plotly figure returned by show_plane_fn()
    """
    arr = image_from_wheel(
        height, width, *ops, wheel=wheel, limits=limits, fix_aspect=False
    )
    return matrix_to_image(arr, format)


def plane_fn_src(
    *ops,
    wheel,
    limits=unit_box,
    height=300,
    width=300,
    fix_aspect=True,
    outfile=None,
    format="png",
    **kwargs,
) -> bytes:
    """Convert a plane function to a formated <img src> string"""
    arr = image_from_wheel(
        height, width, *ops, wheel=wheel, limits=limits, fix_aspect=False
    )
    return matrix_to_src(arr)


def matrix_to_image(arr, format="png"):
    "Convert an RGB array to a PNG image"
    buff = BytesIO()
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(buff, format=format)
    return buff.getvalue()


def matrix_to_src(arr):
    "Convert an RGB array to a string suitable for use in an <img src=> tag"
    src_data = base64.b64encode(matrix_to_image(arr, "png")).decode("utf8")
    return f"data:image/png;base64,{src_data}"


# TODO
def torus_fn(limits=unit_box):
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


def torus(z, limits=unit_box):
    period = limits[1] - limits[0]
    origin = limits[0]

    return (
        np.mod(z.real - origin.real, period.real)
        + 1j * np.mod(z.imag - origin.imag, period.imag)
        + origin
    )


def translate(delta):
    return lambda z: z + delta


def wave_n(n):
    "Sum n plane waves"
    waves = [plane_wave_spacial(np.exp(i * 2j * np.pi / n)) for i in range(n)]
    return lambda z: sum(wave(z) for wave in waves) / n


def plane_wave_spacial(v):
    return lambda z: np.exp(np.pi * 2j * (z * complex(v).conjugate()).real)


def p3_flat(z):
    """Apply P3 in the unit cell (1, ω3) without distortion.

    The asymmetric unit is the pentagon with corners at 0, 1/2, 2/3+ω3/3,
    1/3+2*ω3/3, and ω3/2.
    """
    z = np.atleast_1d(z)
    Z = z.flatten()[:, np.newaxis]

    sq3 = np.sqrt(3)
    w3 = np.exp(2j * np.pi / 3)
    # w12 = np.exp(2j*pi/12)

    # # rotational symmetry around origin
    # r = np.abs(Z)
    # θ = np.arctan2(Z.imag, Z.real)  # radians
    # Z = r*np.exp(1j*(np.mod(θ+pi/3,2*np.pi/3)-pi/3))

    # Wrap to unit cell
    Y = 2 * Z.imag / np.sqrt(3)
    X = Z.real + Z.imag / np.sqrt(3)
    Z = np.mod(X, 1) + w3 * np.mod(Y, 1)
    # Z = np.mod(X+1/2,1)-1/2+w3*(np.mod(Y+1/2,1)-1/2)

    x = Z.real
    y = Z.imag
    Z = np.select(
        [
            (x < 0) & (y - sq3 / 2 > x / 2 - 1 / 4),
            (x >= 0) & (y - sq3 / 2 > -(x + 1 / 2) / 2) & (y > x * sq3),
            (y <= x * sq3) & (y - sq3 / 2 > -(x + 1 / 2) / sq3) & (y > x / 2),
            (y <= x / 2) & (x > 1 / 2),
        ],
        [
            (Z + 1) * w3 + 1,
            (1 - Z) * (1 + w3) - 1,
            w3 * Z + 1,
            (1 - Z) * (1 + w3),
        ],
        Z,
    )
    # # rhombus containing the central hex

    # X = Z.real/np.sqrt(3)+Z.imag
    # Y = Z.imag - Z.real/np.sqrt(3)
    # Z = (np.mod(X+1/2, 1)-1/2)*w12 + (np.mod(Y+1/2,1)-1/2)*w12**5
    # Z.real = np.mod(Z.real+sq3/4,sq3/2)-sq3/4

    return Z.reshape(z.shape)


def wallpaper_hexagonal(a, n, m):
    """Wallpaper groups on a hexagonal lattice

    Functions are represented as fourier coefficients a_nm
    """
    n = np.atleast_1d(n)[np.newaxis, :]
    m = np.atleast_1d(m)[np.newaxis, :]
    a = np.atleast_1d(a)[:, np.newaxis]

    def f(z):
        z = np.atleast_1d(z)
        Z = z.flatten()[:, np.newaxis]
        Y = 2 * Z.imag / np.sqrt(3)
        X = Z.real + Z.imag / np.sqrt(3)
        W = (
            np.exp(2j * np.pi * (n * X + m * Y))
            + np.exp(2j * np.pi * (m * X - (n + m) * Y))
            + np.exp(2j * np.pi * (-(n + m) * X + n * Y))
        ) / 3
        return (W @ a).reshape(z.shape)

    return f


def wallpaper_generic(ω, a, n, m):
    """Wallpaper groups on a generic lattice

    Functions are represented as fourier coefficients a_nm
    """
    n = np.atleast_1d(n)[np.newaxis, :]
    m = np.atleast_1d(m)[np.newaxis, :]
    a = np.atleast_1d(a)[:, np.newaxis]

    def f(z):
        z = np.atleast_1d(z)
        Z = z.flatten()[:, np.newaxis]
        Y = Z.imag / ω.imag
        X = Z.real - ω.real * Y
        W = np.exp(2j * np.pi * (n * X + m * Y))
        return (W @ a).reshape(z.shape)

    return f


def wallpaper_square(a, n, m):
    """Wallpaper groups on a square lattice

    Functions are represented as fourier coefficients a_nm
    """
    n = np.atleast_1d(n)[np.newaxis, :]
    m = np.atleast_1d(m)[np.newaxis, :]
    a = np.atleast_1d(a)[:, np.newaxis]

    def f(z):
        z = np.atleast_1d(z)
        Z = z.flatten()[:, np.newaxis]
        Y = Z.imag
        X = Z.real
        W = (
            np.exp(2j * np.pi * (n * X + m * Y))
            + np.exp(2j * np.pi * (m * X - n * Y))
            + np.exp(2j * np.pi * (-n * X - m * Y))
            + np.exp(2j * np.pi * (-m * X + n * Y))
        ) / 4
        return (W @ a).reshape(z.shape)

    return f


def wallpaper_rhombic(height, a, n, m):
    """Wallpaper groups on a rhombic lattice

    Functions are represented as fourier coefficients a_nm
    """
    n = np.atleast_1d(n)[np.newaxis, :]
    m = np.atleast_1d(m)[np.newaxis, :]
    a = np.atleast_1d(a)[:, np.newaxis]

    def f(z):
        z = np.atleast_1d(z)
        Z = z.flatten()[:, np.newaxis]
        X = Z.real + Z.imag / height / 2
        Y = Z.real - Z.imag / height / 2
        W = (
            np.exp(2j * np.pi * (n * X + m * Y)) + np.exp(2j * np.pi * (m * X + n * Y))
        ) / 2
        return (W @ a).reshape(z.shape)

    return f


def wallpaper_rectangular(height, a, n, m):
    """Wallpaper groups on a rectangular lattice

    Functions are represented as fourier coefficients a_nm
    """
    n = np.atleast_1d(n)[np.newaxis, :]
    m = np.atleast_1d(m)[np.newaxis, :]
    a = np.atleast_1d(a)[:, np.newaxis]

    def f(z):
        z = np.atleast_1d(z)
        Z = z.flatten()[:, np.newaxis]
        Y = Z.imag / height
        X = Z.real
        W = np.exp(2j * np.pi * (n * X + m * Y))
        return (W @ a).reshape(z.shape)

    return f


def wallpaper_pmg(height, coeff: Dict[Tuple[int, int], complex]):
    avg = {}
    for (n, m), a in coeff.items():
        if (n, m) not in avg:
            total = (
                a
                + coeff.get((-n, -m), 0)
                + (-1) ** n * coeff.get((n, -m), 0)
                + (-1) ** n * coeff.get((-n, m), 0)
            ) / 4
            avg[(n, m)] = total
            avg[(-n, -m)] = total
            avg[(n, -m)] = (-1) ** n * total
            avg[(-n, m)] = (-1) ** n * total
    anm = np.empty((len(avg), 3), dtype=complex)
    for i, ((n, m), a) in enumerate(avg.items()):
        anm[i] = (a, n, m)
    # print(anm)
    return wallpaper_generic(height * 1j, anm[:, 0], anm[:, 1], anm[:, 2])


class PreimageMask:
    """Extract the pre-image of a plane function

    To use, create the PreimageMask and chain it after any other operators when applying
    a funtion, eg using `image_from_wheel` or `show_plane_fn`. Later the preimage can be
    extracted from the `mask` property. Values not in the preimage will be True
    (consistent with np.ma.MaskedArray). Visualizing with `masked_wheel` replaces any
    positions not in the preimage with a background color.

    Set the size and limits of the mask to the same as those used to call it for a
    pixel-perfect pre-image. Or, use lower resolution to reduce memory/computation and
    make it easier to visualize fine details.


    ### Example

    ```
    mask = PreimageMask(100, 100, limits=3*unit_box)
    show_plane_fn(
        lambda z: z/abs(z),
        mask,
        wheel=wheel_7
    )
    mask.show_wheel(wheel_7)
    ```


    """

    def __init__(self, height=100, width=100, limits=unit_box):
        self.width = width
        self.height = height
        self.limits = limits
        self.reset()

    def reset(self):
        self.mask = np.ones((self.height, self.width), dtype=bool)
        self.overflow = 0
        self.total = 0

    def __call__(self, z):
        """Marks any points in z as belonging to the preimage.

        Returns z unchanged.
        """
        z = np.atleast_1d(z)
        col = np.floor_divide(
            (z.real - self.limits[0].real) * self.width,
            (self.limits[1].real - self.limits[0].real),
            casting="unsafe",
            dtype=int,
        )
        row = np.floor_divide(
            (z.imag - self.limits[1].imag) * self.height,
            (self.limits[0].imag - self.limits[1].imag),
            casting="unsafe",
            dtype=int,
        )
        # import pdb; pdb.set_trace()
        valid = (0 <= col) & (col < self.width) & (0 <= row) & (row < self.height)
        self.overflow += z.size - np.sum(valid)
        self.total += z.size

        self.mask[row[valid], col[valid]] = False

        return z

    def expand(self, radius: int) -> None:
        """Expand the preimage by radius pixels

        Can be useful for smoothing pixelated-looking images. Reducing the mask
        resolution achieves a similar effect but with a blockier (and faster)
        resullt.
        """
        x, y = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(np.uint8)

        self.mask = ~convolve2d(~self.mask, kernel, mode="same").astype(np.bool_)

    def masked_wheel(self, wheel, background=(0, 0, 0)):
        """Modify a color wheel function by overlaying the preimage as a mask.

        Any colors not in the preimage will be replaced with the backgound color.

        Returns a color wheel function.
        """
        background = np.asarray(background)

        def wrapped(z):
            rgb = wheel(z)
            z = np.atleast_1d(z)
            # grid coords for the z
            col = np.floor_divide(
                (z.real - self.limits[0].real) * self.width,
                (self.limits[1].real - self.limits[0].real),
                casting="unsafe",
                dtype=int,
            )
            row = np.floor_divide(
                (z.imag - self.limits[1].imag) * self.height,
                (self.limits[0].imag - self.limits[1].imag),
                casting="unsafe",
                dtype=int,
            )

            valid = (0 <= col) & (col < self.width) & (0 <= row) & (row < self.height)
            masked = self.mask[row[valid], col[valid]]
            valid[valid] = ~masked
            rgb[~valid, :] = background

            return rgb

        return wrapped

    def show_wheel(self, wheel, background=(0, 0, 0)):
        """Display the specified color wheel by overlaying the preimage as a mask.

        Any colors not in the preimage will be replaced with the backgound color.
        """
        return show_plane_fn(
            wheel=self.masked_wheel(wheel, background=background), limits=self.limits
        )
