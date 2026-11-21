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

import math

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


# ---------------------------------------------------------------------------
# Validation and shared geometry helpers
# ---------------------------------------------------------------------------


def _validate_plane(plane, allow_empty=False):
    """Return ``(height, width)`` of a rectangular finite plane.

    Every sample must be a finite real ``int``/``float`` (booleans are
    rejected).  Ragged rows, non-list structures, empty rows and, unless
    ``allow_empty`` is set, an empty plane raise :class:`ChromaError`.
    """
    if not isinstance(plane, list):
        raise ChromaError("plane must be a list of rows")
    if len(plane) == 0:
        if allow_empty:
            return 0, 0
        raise ChromaError("plane must not be empty")
    width = None
    for row in plane:
        if not isinstance(row, list):
            raise ChromaError("every plane row must be a list")
        if width is None:
            width = len(row)
            if width == 0:
                raise ChromaError("plane rows must not be empty")
        elif len(row) != width:
            raise ChromaError("plane rows must all have the same length")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ChromaError("plane samples must be numeric")
            if not math.isfinite(value):
                raise ChromaError("plane samples must be finite")
    return len(plane), width


def _classify(h, w, ch, cw):
    """Classify a chroma shape against the luma shape ``h x w``."""
    if (ch, cw) == (h, w):
        return "444"
    if (ch, cw) == (h, (w + 1) // 2):
        return "422"
    if (ch, cw) == ((h + 1) // 2, (w + 1) // 2):
        return "420"
    raise ChromaError(
        f"chroma plane shape {ch}x{cw} matches no layout for luma {h}x{w}"
    )


def _sample_linear(grid, n, position):
    """Linearly sample a 1-D grid at float ``position`` with edge clamping."""
    if position <= 0.0:
        return grid[0]
    if position >= n - 1:
        return grid[n - 1]
    index = int(math.floor(position))
    frac = position - index
    return grid[index] * (1.0 - frac) + grid[index + 1] * frac


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


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
    h, w = _validate_plane(plane, allow_empty=True)
    if h == 0:
        return []
    out_h = (h + 1) // 2
    out_w = (w + 1) // 2
    result = []
    for cy in range(out_h):
        y0 = 2 * cy
        y1 = min(y0 + 1, h - 1)
        out_row = []
        for cx in range(out_w):
            x0 = 2 * cx
            x1 = min(x0 + 1, w - 1)
            total = (
                plane[y0][x0]
                + plane[y0][x1]
                + plane[y1][x0]
                + plane[y1][x1]
            )
            out_row.append(total / 4.0)
        result.append(out_row)
    return result


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
    h, w = _validate_plane(plane, allow_empty=True)
    if h == 0:
        return []
    return [[plane[y][x] for x in range(0, w, 2)] for y in range(h)]


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------


def _check_target(w, h):
    if isinstance(w, bool) or isinstance(h, bool) or not isinstance(w, int) or not isinstance(h, int):
        raise ChromaError("target dimensions must be integers")
    if w < 0 or h < 0:
        raise ChromaError("target dimensions must not be negative")
    return w == 0 or h == 0


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
    if _check_target(w, h):
        return []
    ch, cw = _validate_plane(plane)
    # Horizontal bilinear pass once per source row, then a vertical pass over
    # the column-interpolated values, so each source row is sampled only once.
    horizontal = [
        [
            _sample_linear(plane[sy], cw, (x + 0.5) * cw / w - 0.5)
            for x in range(w)
        ]
        for sy in range(ch)
    ]
    result = []
    for y in range(h):
        py = (y + 0.5) * ch / h - 0.5
        if py <= 0.0:
            result.append(horizontal[0][:])
        elif py >= ch - 1:
            result.append(horizontal[ch - 1][:])
        else:
            iy = int(math.floor(py))
            fy = py - iy
            top, bottom = horizontal[iy], horizontal[iy + 1]
            result.append(
                [
                    top[x] * (1.0 - fy) + bottom[x] * fy
                    for x in range(w)
                ]
            )
    return result


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
    if _check_target(w, h):
        return []
    ch, cw = _validate_plane(plane)
    result = []
    for y in range(h):
        sy = min(ch - 1, int(math.floor(y * ch / h)))
        source_row = plane[sy]
        result.append(
            [
                _sample_linear(source_row, cw, (x + 0.5) * cw / w - 0.5)
                for x in range(w)
            ]
        )
    return result


# ---------------------------------------------------------------------------
# YUV / RGB matrices
# ---------------------------------------------------------------------------

# Forward YUV -> RGB coefficients keyed by matrix name.  Each tuple holds
# (R_v, G_u, G_v, B_u); with U = u - 0.5 and V = v - 0.5:
#   R = Y + R_v * V
#   G = Y + G_u * U + G_v * V
#   B = Y + B_u * U
_MATRIX_TO_RGB = {
    "bt601": (1.402, -0.344136, -0.714136, 1.772),
    "bt709": (1.5748, -0.187324, -0.468124, 1.8556),
}

# Inverse RGB -> full-range YUV coefficients (exact algebraic inverses of the
# forward matrices, rounded to the standard six-decimal constants).
_MATRIX_TO_YUV = {
    "bt601": {
        "yr": 0.299, "yg": 0.587, "yb": 0.114,
        "ur": -0.168736, "ug": -0.331264, "ub": 0.5,
        "vr": 0.5, "vg": -0.418688, "vb": -0.081312,
    },
    "bt709": {
        "yr": 0.2126, "yg": 0.7152, "yb": 0.0722,
        "ur": -0.114572, "ug": -0.385428, "ub": 0.5,
        "vr": 0.5, "vg": -0.454153, "vb": -0.045847,
    },
}


def _expand_chroma(plane, h, w, ch, cw, layout):
    """Nearest-neighbour replication of a chroma plane to ``h x w``."""
    expanded = []
    for y in range(h):
        if layout == "444":
            sy = y
        else:
            sy = min(ch - 1, int(math.floor(y * ch / h)))
        out_row = []
        for x in range(w):
            if layout == "444":
                sx = x
            else:
                sx = min(cw - 1, int(math.floor(x * cw / w)))
            out_row.append(plane[sy][sx])
        expanded.append(out_row)
    return expanded


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
    h, w = _validate_plane(y)
    ch, cw = _validate_plane(u)
    vh, vw = _validate_plane(v)
    if (vh, vw) != (ch, cw):
        raise ChromaError("u and v planes must have the same shape")
    layout = _classify(h, w, ch, cw)
    if matrix not in _MATRIX_TO_RGB:
        raise ChromaError(f"unknown matrix: {matrix!r}")
    r_v, g_u, g_v, b_u = _MATRIX_TO_RGB[matrix]

    u_full = _expand_chroma(u, h, w, ch, cw, layout)
    v_full = _expand_chroma(v, h, w, ch, cw, layout)

    r_plane, g_plane, b_plane = [], [], []
    for yy in range(h):
        r_row, g_row, b_row = [], [], []
        for xx in range(w):
            yy_val = y[yy][xx]
            uu = u_full[yy][xx] - 0.5
            vv = v_full[yy][xx] - 0.5
            r_row.append(yy_val + r_v * vv)
            g_row.append(yy_val + g_u * uu + g_v * vv)
            b_row.append(yy_val + b_u * uu)
        r_plane.append(r_row)
        g_plane.append(g_row)
        b_plane.append(b_row)
    return {"r": r_plane, "g": g_plane, "b": b_plane}


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
    ``C = 128/255 + (C_full - 0.5) * 224/255``; black therefore maps to
    ``16/255`` and neutral chroma to ``128/255`` rather than being lifted to
    full-range zero.  The return value is ``{"y": plane, "u": plane,
    "v": plane}``.

    Unknown matrices, unknown range modes, mismatched plane shapes, ragged
    input, or non-finite samples raise :class:`ChromaError`.  Results are
    checked at an absolute tolerance of ``1e-6``.
    """
    h, w = _validate_plane(r)
    gh, gw = _validate_plane(g)
    bh, bw = _validate_plane(b)
    if (gh, gw) != (h, w) or (bh, bw) != (h, w):
        raise ChromaError("r, g and b planes must have the same shape")
    if matrix not in _MATRIX_TO_YUV:
        raise ChromaError(f"unknown matrix: {matrix!r}")
    if range_mode not in ("full", "limited"):
        raise ChromaError(f"unknown range mode: {range_mode!r}")
    coeff = _MATRIX_TO_YUV[matrix]

    y_plane, u_plane, v_plane = [], [], []
    for yy in range(h):
        y_row, u_row, v_row = [], [], []
        for xx in range(w):
            rr = r[yy][xx]
            gg = g[yy][xx]
            bb = b[yy][xx]
            y_val = coeff["yr"] * rr + coeff["yg"] * gg + coeff["yb"] * bb
            u_val = (
                coeff["ur"] * rr + coeff["ug"] * gg + coeff["ub"] * bb + 0.5
            )
            v_val = (
                coeff["vr"] * rr + coeff["vg"] * gg + coeff["vb"] * bb + 0.5
            )
            if range_mode == "limited":
                y_val = 16.0 / 255.0 + y_val * 219.0 / 255.0
                u_val = 128.0 / 255.0 + (u_val - 0.5) * 224.0 / 255.0
                v_val = 128.0 / 255.0 + (v_val - 0.5) * 224.0 / 255.0
            y_row.append(y_val)
            u_row.append(u_val)
            v_row.append(v_val)
        y_plane.append(y_row)
        u_plane.append(u_row)
        v_plane.append(v_row)
    return {"y": y_plane, "u": u_plane, "v": v_plane}


# ---------------------------------------------------------------------------
# Plane layout validation and batch conversion
# ---------------------------------------------------------------------------


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
    h, w = _validate_plane(y)
    ch, cw = _validate_plane(u)
    vh, vw = _validate_plane(v)
    if (vh, vw) != (ch, cw):
        raise ChromaError("u and v planes must have the same shape")
    layout = _classify(h, w, ch, cw)
    return {"format": layout, "y": y, "u": u, "v": v}


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
    if format not in ("420", "422"):
        raise ChromaError(f"unsupported chroma format: {format!r}")
    if not isinstance(images, list):
        raise ChromaError("images must be a list of frame mappings")
    if len(images) == 0:
        return {"format": format, "frames": [], "count": 0}

    # Phase 1: validate every frame before constructing any output, so a
    # malformed frame fails the whole batch atomically.
    validated = []
    for frame in images:
        if not isinstance(frame, dict) or not all(
            key in frame for key in ("y", "u", "v")
        ):
            raise ChromaError("each frame must contain y, u and v planes")
        h, w = _validate_plane(frame["y"])
        uh, uw = _validate_plane(frame["u"])
        vh, vw = _validate_plane(frame["v"])
        if (uh, uw) != (h, w) or (vh, vw) != (h, w):
            raise ChromaError("frame planes must share the same 4:4:4 shape")
        validated.append((frame, h, w))

    # Phase 2: build fully independent output planes.  Luma is deep-copied and
    # the downsamplers allocate fresh lists, so no buffers are shared.
    downsample = downsample_420 if format == "420" else downsample_422
    frames = []
    for frame, _h, _w in validated:
        frames.append(
            {
                "y": [row[:] for row in frame["y"]],
                "u": downsample(frame["u"]),
                "v": downsample(frame["v"]),
            }
        )
    return {"format": format, "frames": frames, "count": len(frames)}
