"""Shared test fixtures.

Synthetic NIfTI volumes are written to a tmp dir per-test so the suite stays
fast, deterministic, and free of committed binaries. The factory mirrors the
shapes `load_stat_map` must handle: 3D vs 4D, and neurological vs radiological
affines (the sign of affine[0, 0] drives the L/R flip).
"""

import numpy as np
import nibabel as nib
import pytest

# Neurological orientation: +x voxel direction maps to +x (right) in mm.
NEUROLOGICAL_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float64)

# Radiological orientation: +x voxel direction maps to -x (left) in mm.
RADIOLOGICAL_AFFINE = np.diag([-2.0, 2.0, 2.0, 1.0]).astype(np.float64)


class FakeLabeler:
    """A RegionLabeler backed by an explicit {(i, j, k): name} map."""

    def __init__(self, labels: dict[tuple[int, int, int], str]):
        self._labels = labels

    def label_at(self, vox: tuple[int, int, int]) -> str:
        return self._labels.get(tuple(vox), "Unknown")


@pytest.fixture
def make_nii(tmp_path):
    """Return a factory that writes a synthetic .nii.gz and returns its path."""

    def _make(data, affine=NEUROLOGICAL_AFFINE, name="stat.nii.gz"):
        img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
        path = tmp_path / name
        nib.save(img, str(path))
        return path

    return _make
