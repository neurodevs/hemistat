"""Tests for hemistat.analysis geometry helpers.

Written first (RED): hemistat.analysis does not exist yet.

`vox_to_mni(affine, idx, axis)` converts a voxel slice index along one axis to
its MNI mm coordinate. It assumes a diagonal (axis-aligned) affine — the
invariant the pipeline guarantees by resampling to template space — so the
result is just `affine[axis, axis] * idx + affine[axis, 3]`, independent of the
off-axis voxel coords (and thus of volume shape).
"""

import numpy as np
import pytest

from hemistat.analysis import vox_to_mni

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
