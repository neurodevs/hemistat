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

from hemistat.analysis import active_slices, analyze_stat_map, vox_to_mni
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
    sm = StatMap(path=Path("t.nii.gz"), data=data)

    analysis = analyze_stat_map(sm)

    assert {
        "axial": analysis.axial,
        "coronal": analysis.coronal,
        "sagittal": analysis.sagittal,
    } == {"axial": [1], "coronal": [2], "sagittal": [3]}
