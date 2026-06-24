"""Tests for hemistat.analysis geometry helpers.

Written first (RED): hemistat.analysis does not exist yet.

`vox_to_mni(affine, idx, axis)` converts a voxel slice index along one axis to
its MNI mm coordinate. It assumes a diagonal (axis-aligned) affine — the
invariant the pipeline guarantees by resampling to template space — so the
result is just `affine[axis, axis] * idx + affine[axis, 3]`, independent of the
off-axis voxel coords (and thus of volume shape).
"""

from pathlib import Path

import numpy as np
import pytest

from hemistat.analysis import (
    active_slices,
    analyze_stat_map,
    lateralization_score,
    mirror_pairs,
    split_hemispheres,
    vox_to_mni,
)
from hemistat.io import StatMap

# Typical MNI152 2mm affine: 2mm isotropic voxels, axis-aligned (diagonal).
MNI_2MM_AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.mark.parametrize(
    "axis, idx, expected_mm",
    [
        (0, 5, -80.0),    # 2 * 5  - 90
        (1, 10, -106.0),  # 2 * 10 - 126
        (2, 3, -66.0),    # 2 * 3  - 72
    ],
)
def test_maps_slice_index_to_mm(axis, idx, expected_mm):
    assert vox_to_mni(MNI_2MM_AFFINE, idx, axis) == expected_mm


def test_active_slices_returns_indices_with_nonzero_voxels():
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[:, :, 1] = 2.0   # axis-2 slice 1 active
    data[:, :, 3] = -1.0  # axis-2 slice 3 active (negatives count too)

    assert active_slices(data, axis=2) == [1, 3]


def test_analyze_collects_active_slices_per_axis():
    # A single hot voxel is active on exactly one slice of each axis.
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[3, 2, 1] = 1.0
    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=np.eye(4))

    analysis = analyze_stat_map(sm)

    assert {
        "axial": analysis.axial,
        "coronal": analysis.coronal,
        "sagittal": analysis.sagittal,
    } == {"axial": [1], "coronal": [2], "sagittal": [3]}


def test_analyze_includes_mirror_pairs():
    # affine: mni_x = 2*i - 3  ->  geometric mirror indices: 0<->3, 1<->2.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0
    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=affine)

    analysis = analyze_stat_map(sm)

    assert analysis.mirror == [(1, 2)]


def test_split_hemispheres_partitions_voxels_on_mni_x():
    # affine: mni_x = 2*i - 3  ->  i=0,1 are left (x<=0); i=2,3 are right (x>0).
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0   # left hemisphere
    data[3, 0, 0] = 7.0   # right hemisphere

    left, right = split_hemispheres(data, affine)

    expected_left = np.zeros((4, 1, 1), dtype=np.float32)
    expected_left[1, 0, 0] = 5.0
    expected_right = np.zeros((4, 1, 1), dtype=np.float32)
    expected_right[3, 0, 0] = 7.0

    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)


def test_mirror_pairs_to_geometric_mirror_even_when_blank():
    # affine: mni_x = 2*i - 3  ->  geometric mirror indices: 0<->3, 1<->2.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0   # active on the left; its mirror (i=2) is blank

    # Pairs to geometric mirror 2 despite slice 2 having no activation.
    assert mirror_pairs(data, affine, axis=0) == [(1, 2)]


def test_mirror_pairs_returns_none_when_mirror_off_grid():
    # affine: mni_x = 2*i + 2  ->  every voxel is positive x, so mirrors are off-grid.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0   # mirror coord is -4 mm -> index -3, off the grid

    assert mirror_pairs(data, affine, axis=0) == [(1, None)]


def test_lateralization_score_is_one_when_fully_lateralized():
    # Active slice 1 whose mirror (slice 2) is blank -> all activation is unique.
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0
    pairs = [(1, 2)]

    assert lateralization_score(data, pairs) == 1.0


def test_lateralization_score_treats_none_mirror_as_fully_unique():
    # mirror_pairs yields None when the mirror is off-grid: no counterpart
    # exists, so all of the slice's activation is unique.
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0
    pairs = [(1, None)]

    assert lateralization_score(data, pairs) == 1.0


def test_lateralization_score_is_zero_when_no_pairs():
    # No active slices -> nothing to measure -> 0.0 (not NaN).
    data = np.zeros((4, 1, 1), dtype=np.float32)

    assert lateralization_score(data, []) == 0.0
