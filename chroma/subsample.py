"""Chroma subsampling and YUV conversion contracts.

Planes are rectangular, row-major ``list[list[float]]`` values addressed as
``plane[y][x]`` (rows first).  Luma is normalized to ``[0.0, 1.0]``.  Chroma
is also normalized to ``[0.0, 1.0]`` with neutral at ``0.5``; the full-range
YUV equations operate on those normalized values directly.

Alignment for a luma plane of size ``h x w``:

* 4:4:4 -- chroma is ``h x w``.
* 4:2:2 -- chroma is ``h x ceil(w / 2)``; vertical resolution is preserved.
* 4:2:0 -- chroma is ``ceil(h / 2) x ceil(w / 2)``.

Chroma samples are centred in their support.  Upsampling therefore maps an
output luma sample centre ``(x + 0.5, y + 0.5)`` into the chroma index space
as ``((x + 0.5) * cw / w - 0.5, (y + 0.5) * ch / h - 0.5)``; bilinear weights
and edge clamping are applied there.  This keeps chroma aligned instead of
shifting it by half a luma pixel.

All numeric results are compared with an absolute tolerance of ``1e-6``.
Invalid, ragged, or non-finite planes, unsupported formats or matrices, and
inconsistent plane sizes raise :class:`ChromaError`.
"""

from __future__ import annotations

__all__ = [
    "ChromaError",
    "align_planes",
    "downsample_420",
    "downsample_422",
    "rgb_to_yuv",
    "subsample_batch",
    "upsample_420",
    "upsample_422",
    "yuv_to_rgb",
]


class ChromaError(ValueError):
    """Raised for invalid planes, formats, matrices, or size mismatches."""


def downsample_420(plane: list[list[float]]) -> list[list[float]]:
    """Downsample a full-resolution plane to 4:2:0.

    Every output sample is the arithmetic mean of its 2x2 luma block.  An odd
    width or height is padded by replicating the last column or row, so edge
    blocks average real samples twice rather than introducing zeros.  The
    result has ``ceil(h / 2)`` rows and ``ceil(w / 2)`` columns.  An empty
    plane maps to ``[]``.

    Input is a rectangular, row-major ``list[list[float]]``; ragged rows or
    non-finite samples raise :class:`ChromaError`.  Results are checked at an
    absolute tolerance of ``1e-6``.
    """
    raise NotImplementedError


def downsample_422(plane: list[list[float]]) -> list[list[float]]:
    """Downsample a full-resolution plane to 4:2:2.

    The operation is horizontal 2:1 decimation: the left sample of each
    horizontal pair is kept and the right sample is dropped (this is not an
    average).  Vertical resolution is unchanged, and an odd width keeps the
    final unpaired sample, so the result is ``h x ceil(w / 2)``.

    Input is a rectangular, row-major ``list[list[float]]``; an empty plane
    maps to ``[]``.  Ragged rows or non-finite samples raise
    :class:`ChromaError`.  Results are checked at an absolute tolerance of
    ``1e-6``.
    """
    raise NotImplementedError


def upsample_420(plane: list[list[float]], w: int, h: int) -> list[list[float]]:
    """Bilinearly upsample a 4:2:0 chroma plane to exactly ``h x w`` samples.

    ``plane`` is a ``ch x cw`` chroma grid.  Output sample centres are mapped
    into chroma index space with the centre-aligned transform
    ``x' = (x + 0.5) * cw / w - 0.5`` and
    ``y' = (y + 0.5) * ch / h - 0.5``; bilinear weights are applied after
    clamping ``x'``/``y'`` to the valid index range.  This is not nearest
    neighbour and does not shift chroma by a half pixel.

    The returned plane has exactly ``h`` rows of exactly ``w`` values.  If
    either target dimension is zero the result is ``[]``.  An empty source
    plane with a positive target, a negative target, ragged input, or
    non-finite samples raises :class:`ChromaError`.  Results are checked at an
    absolute tolerance of ``1e-6``.
    """
    raise NotImplementedError


def upsample_422(plane: list[list[float]], w: int, h: int) -> list[list[float]]:
    """Bilinearly upsample a 4:2:2 chroma plane to exactly ``h x w`` samples.

    Interpolation is horizontal only: output columns use the centre-aligned
    transform ``x' = (x + 0.5) * cw / w - 0.5`` with bilinear weights and edge
    clamping.  Rows are not interpolated; output row ``y`` reads source row
    ``min(ch - 1, floor(y * ch / h))``, which is the identity when
    ``h == ch``.  This preserves the full vertical chroma resolution of
    4:2:2 while avoiding a horizontal half-pixel shift.

    The returned plane has exactly ``h`` rows of exactly ``w`` values.  If
    either target dimension is zero the result is ``[]``.  An empty source
    plane with a positive target, a negative target, ragged input, or
    non-finite samples raises :class:`ChromaError`.  Results are checked at an
    absolute tolerance of ``1e-6``.
    """
    raise NotImplementedError


def yuv_to_rgb(
    y: list[list[float]],
    u: list[list[float]],
    v: list[list[float]],
    matrix: str,
) -> dict:
    """Convert normalized full-range YUV planes to RGB planes.

    ``y`` is an ``h x w`` luma plane.  ``u`` and ``v`` may be ``h x w``
    (4:4:4), ``h x ceil(w / 2)`` (4:2:2), or ``ceil(h / 2) x ceil(w / 2)``
    (4:2:0); chroma is expanded to luma resolution with nearest-neighbour
    replication before the matrix is applied.  All three planes use values in
    ``[0.0, 1.0]`` with neutral chroma at ``0.5``.

    ``matrix`` is ``"bt601"`` or ``"bt709"``.  With ``U = u - 0.5`` and
    ``V = v - 0.5`` the coefficients are:

    * ``bt601``: ``R = Y + 1.402 V``;
      ``G = Y - 0.344136 U - 0.714136 V``; ``B = Y + 1.772 U``.
    * ``bt709``: ``R = Y + 1.5748 V``;
      ``G = Y - 0.187324 U - 0.468124 V``; ``B = Y + 1.8556 U``.

    The return value is ``{"r": plane, "g": plane, "b": plane}`` with the
    same ``h x w`` shape as ``y``.  Values are floats and are not clamped, so
    out-of-gamut results remain visible.  An unknown ``matrix`` raises
    :class:`ChromaError` instead of falling back to BT.601.  Results are
    checked at an absolute tolerance of ``1e-6``.
    """
    raise NotImplementedError


def rgb_to_yuv(
    r: list[list[float]],
    g: list[list[float]],
    b: list[list[float]],
    matrix: str,
    range_mode: str = "full",
) -> dict:
    """Convert normalized RGB planes to YUV planes.

    This is the inverse of :func:`yuv_to_rgb` for full range.  The three input
    planes must have the same rectangular ``h x w`` shape and values in
    ``[0.0, 1.0]``.  ``matrix`` is ``"bt601"`` or ``"bt709"`` and selects the
    same coefficients as :func:`yuv_to_rgb`.

    ``range_mode`` is ``"full"`` or ``"limited"``.  Full range keeps ``Y`` in
    ``[0.0, 1.0]`` with neutral chroma at ``0.5``.  Limited range encodes
    ``Y = 16/255 + Y_full * 219/255`` and
    ``U = V = 16/255 + C_full * 224/255``; black therefore maps to ``16/255``
    and neutral chroma to ``128/255`` rather than being lifted to full-range
    zero.  The return value is ``{"y": plane, "u": plane, "v": plane}``.

    Unknown matrices, unknown range modes, mismatched plane shapes, ragged
    input, or non-finite samples raise :class:`ChromaError`.  Results are
    checked at an absolute tolerance of ``1e-6``.
    """
    raise NotImplementedError


def align_planes(
    y: list[list[float]],
    u: list[list[float]],
    v: list[list[float]],
) -> dict:
    """Validate and classify a YUV plane triple by its sampling layout.

    ``y`` is an ``h x w`` luma plane.  ``u`` and ``v`` must have identical
    shapes and match exactly one supported layout:

    * ``"444"``: ``h x w``.
    * ``"422"``: ``h x ceil(w / 2)``.
    * ``"420"``: ``ceil(h / 2) x ceil(w / 2)``.

    Exact equality with the luma shape is classified as ``"444"`` first.  The
    return value is ``{"format": format, "y": y, "u": u, "v": v}`` with the
    validated planes passed through.

    A mismatched u/v shape, a shape that matches no layout, an empty luma
    plane, or ragged input raises :class:`ChromaError` rather than allowing a
    later out-of-bounds read.  Results are checked at an absolute tolerance of
    ``1e-6``.
    """
    raise NotImplementedError


def subsample_batch(images: list[dict], format: str) -> dict:
    """Convert a batch of 4:4:4 frames to one chroma format atomically.

    ``images`` is a list of mappings, each containing full-resolution
    rectangular ``"y"``, ``"u"`` and ``"v"`` planes of the same ``h x w``
    shape.  ``format`` is ``"420"`` or ``"422"``.  The luma plane is kept
    unchanged; u and v are reduced with :func:`downsample_420` or
    :func:`downsample_422` respectively.

    The return value is ``{"format": format, "frames": [...], "count": n}``,
    where each output frame is a fresh ``{"y": ..., "u": ..., "v": ...}``
    mapping in the input order.  An empty batch returns
    ``{"format": format, "frames": [], "count": 0}``.

    Conversion is all-or-nothing: every frame is validated before any result
    is built, and a malformed frame raises :class:`ChromaError` with no
    partial result returned.  Output planes never share buffers with each
    other or with the input, so mutating one frame cannot corrupt another and
    the input frames are never mutated.  Results are checked at an absolute
    tolerance of ``1e-6``.
    """
    raise NotImplementedError
