"""Red tests for the chroma subsampling and YUV conversion kernels.

The tests are intentionally read-only: fix ``chroma/subsample.py`` instead of
editing this file.  Every test currently fails with ``NotImplementedError``
from one of the eight stubs.
"""

import pytest

from chroma.subsample import (
    ChromaError,
    align_planes,
    downsample_420,
    downsample_422,
    rgb_to_yuv,
    subsample_batch,
    upsample_420,
    upsample_422,
    yuv_to_rgb,
)


def _constant(h, w, value):
    return [[float(value)] * w for _ in range(h)]


def _assert_plane_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for row, expected_row in zip(actual, expected):
        assert len(row) == len(expected_row)
        assert row == pytest.approx(expected_row, abs=tol)


def test_downsample_420_averages_each_2x2_block():
    source = [[0.0, 100.0], [100.0, 0.0]]

    result = downsample_420(source)

    assert len(result) == 1
    assert result[0] == pytest.approx([50.0], abs=1e-6)


def test_downsample_420_odd_edges_replicate_instead_of_zero():
    source = [
        [10.0, 20.0, 30.0],
        [30.0, 40.0, 50.0],
        [50.0, 60.0, 70.0],
    ]

    result = downsample_420(source)

    _assert_plane_close(result, [[25.0, 40.0], [55.0, 70.0]])


def test_upsample_420_is_bilinear_not_nearest():
    source = [[0.0, 100.0], [100.0, 0.0]]

    result = upsample_420(source, 4, 4)

    _assert_plane_close(
        result,
        [
            [0.0, 25.0, 75.0, 100.0],
            [25.0, 37.5, 62.5, 75.0],
            [75.0, 62.5, 37.5, 25.0],
            [100.0, 75.0, 25.0, 0.0],
        ],
    )


def test_upsample_output_size_is_exact():
    source = [[0.0, 100.0], [100.0, 0.0]]

    wide = upsample_420(source, 5, 3)
    tall = upsample_422(source, 5, 2)

    assert len(wide) == 3
    assert all(len(row) == 5 for row in wide)
    assert len(tall) == 2
    assert all(len(row) == 5 for row in tall)
    assert wide[0][0] == pytest.approx(0.0, abs=1e-6)
    _assert_plane_close(
        tall,
        [
            [0.0, 10.0, 50.0, 90.0, 100.0],
            [100.0, 90.0, 50.0, 10.0, 0.0],
        ],
    )


def test_yuv_to_rgb_uses_bt709_coefficients():
    y = _constant(2, 2, 0.5)
    u = [[0.5]]
    v = [[1.0]]

    result = yuv_to_rgb(y, u, v, "bt709")

    assert len(result["r"]) == 2
    assert len(result["r"][0]) == 2
    for row in range(2):
        for col in range(2):
            assert result["r"][row][col] == pytest.approx(1.2874, abs=1e-6)
            assert result["g"][row][col] == pytest.approx(0.265938, abs=1e-6)
            assert result["b"][row][col] == pytest.approx(0.5, abs=1e-6)
    assert abs(result["r"][0][0] - 1.201) > 1e-3


def test_rgb_to_yuv_distinguishes_limited_range():
    black = ([[0.0]], [[0.0]], [[0.0]])
    white = ([[1.0]], [[1.0]], [[1.0]])

    limited_black = rgb_to_yuv(*black, "bt709", range_mode="limited")
    limited_white = rgb_to_yuv(*white, "bt709", range_mode="limited")
    full_black = rgb_to_yuv(*black, "bt709", range_mode="full")

    assert limited_black["y"][0][0] == pytest.approx(16 / 255, abs=1e-6)
    assert limited_black["u"][0][0] == pytest.approx(128 / 255, abs=1e-6)
    assert limited_black["v"][0][0] == pytest.approx(128 / 255, abs=1e-6)
    assert limited_white["y"][0][0] == pytest.approx(235 / 255, abs=1e-6)
    assert full_black["y"][0][0] == pytest.approx(0.0, abs=1e-6)
    assert full_black["u"][0][0] == pytest.approx(0.5, abs=1e-6)
    assert full_black["v"][0][0] == pytest.approx(0.5, abs=1e-6)


def test_upsample_420_chroma_stays_center_aligned():
    source = [[0.0, 100.0]]

    result = upsample_420(source, 4, 1)

    _assert_plane_close(result, [[0.0, 25.0, 75.0, 100.0]])


def test_align_planes_rejects_mismatched_sizes():
    y = _constant(4, 4, 128.0)
    u = _constant(2, 2, 128.0)
    v = _constant(1, 2, 128.0)

    with pytest.raises(ChromaError):
        align_planes(y, u, v)

    valid = align_planes(y, u, u)

    assert valid["format"] == "420"
    assert valid["y"][0][0] == pytest.approx(128.0, abs=1e-6)


def test_subsample_batch_keeps_frame_buffers_independent():
    frame_a = {
        "y": _constant(4, 4, 10.0),
        "u": _constant(4, 4, 20.0),
        "v": _constant(4, 4, 30.0),
    }
    frame_b = {
        "y": _constant(4, 4, 100.0),
        "u": _constant(4, 4, 200.0),
        "v": _constant(4, 4, 250.0),
    }

    result = subsample_batch([frame_a, frame_b], "420")
    frames = result["frames"]

    assert len(frames) == 2
    assert frames[0]["u"][0][0] == pytest.approx(20.0, abs=1e-6)
    assert frames[1]["u"][0][0] == pytest.approx(200.0, abs=1e-6)
    frames[0]["u"][0][0] = -1.0
    assert frames[1]["u"][0][0] == pytest.approx(200.0, abs=1e-6)
    assert frame_a["u"][0][0] == pytest.approx(20.0, abs=1e-6)

    bad_frame = {
        "y": _constant(4, 4, 0.0),
        "u": _constant(2, 2, 0.0),
        "v": _constant(4, 4, 0.0),
    }
    with pytest.raises(ChromaError):
        subsample_batch([frame_a, bad_frame], "420")


def test_unknown_matrix_raises_chroma_error():
    with pytest.raises(ChromaError):
        yuv_to_rgb([[0.5]], [[0.5]], [[0.5]], "not-a-matrix")


def test_yuv_rgb_yuv_roundtrip_within_tolerance():
    y = [[0.6, 0.5], [0.55, 0.7]]
    u = [[0.45, 0.55], [0.5, 0.48]]
    v = [[0.52, 0.47], [0.5, 0.55]]

    rgb = yuv_to_rgb(y, u, v, "bt709")
    back = rgb_to_yuv(rgb["r"], rgb["g"], rgb["b"], "bt709", range_mode="full")

    _assert_plane_close(back["y"], y)
    _assert_plane_close(back["u"], u)
    _assert_plane_close(back["v"], v)


def test_subsample_same_format_is_idempotent():
    images = [
        {
            "y": _constant(4, 4, 10.0),
            "u": _constant(4, 4, 20.0),
            "v": _constant(4, 4, 30.0),
        },
        {
            "y": _constant(4, 4, 40.0),
            "u": _constant(4, 4, 50.0),
            "v": _constant(4, 4, 60.0),
        },
    ]

    first = subsample_batch(images, "420")
    second = subsample_batch(images, "420")

    assert first == second
    assert images[0]["u"][0][0] == pytest.approx(20.0, abs=1e-6)
    assert images[1]["v"][3][3] == pytest.approx(60.0, abs=1e-6)


def test_bt2020_matrix_raises_chroma_error():
    with pytest.raises(ChromaError):
        yuv_to_rgb([[0.5]], [[0.5]], [[0.5]], "bt2020")


def test_empty_plane_returns_empty():
    assert downsample_420([]) == []
    assert downsample_422([]) == []
    assert upsample_420([], 0, 0) == []


def test_batch_success_preserves_order():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    images = [
        {
            "y": _constant(4, 4, value),
            "u": [
                [value + 1.0, value + 2.0, value + 3.0, value + 4.0]
                for _ in range(4)
            ],
            "v": [
                [value + 5.0, value + 6.0, value + 7.0, value + 8.0]
                for _ in range(4)
            ],
        }
        for value in values
    ]

    result = subsample_batch(images, "422")
    frames = result["frames"]

    assert result["count"] == 5
    assert result["format"] == "422"
    assert [frame["y"][0][0] for frame in frames] == pytest.approx(
        values, abs=1e-6
    )
    assert all(len(frame["y"]) == 4 and len(frame["y"][0]) == 4 for frame in frames)
    assert all(len(frame["u"]) == 4 and len(frame["u"][0]) == 2 for frame in frames)
    _assert_plane_close(frames[0]["u"], [[11.0, 13.0]] * 4)
    _assert_plane_close(frames[0]["v"], [[15.0, 17.0]] * 4)
