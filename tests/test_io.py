"""Tests for hemistat.io.load_stat_map — the entry point where real NIfTI
data enters the pipeline.

These are written first (RED): hemistat.io does not exist yet, so the import
below fails. Implementing `load_stat_map`/`StatMap` to satisfy these is the
next TDD step.
"""

import numpy as np
import pytest

from hemistat.io import load_stat_map, StatMap
from tests.conftest import RADIOLOGICAL_AFFINE

@pytest.fixture(autouse=True)
def before_each():
    yield


def test_returns_statmap_instance(make_nii):
    path = make_nii(np.zeros((4, 5, 6)), name="example.nii.gz")
    sm = load_stat_map(path)

    assert isinstance(sm, StatMap)

def test_includes_path_in_statmap(make_nii):
    path = make_nii(np.zeros((4, 5, 6)), name="example.nii.gz")
    sm = load_stat_map(path)
    
    assert sm.path == path

def test_includes_affine_in_statmap(make_nii):
    path = make_nii(np.zeros((2, 2, 2)), affine=RADIOLOGICAL_AFFINE)
    sm = load_stat_map(path)

    assert sm.affine is not None
    np.testing.assert_allclose(sm.affine, RADIOLOGICAL_AFFINE)


def test_data_is_3d_float32_with_values_preserved(make_nii):
    data = np.zeros((4, 5, 6), dtype=np.float32)
    data[1, 2, 3] = 5.0
    path = make_nii(data, name="example.nii.gz")

    sm = load_stat_map(path)

    assert sm.data.shape == (4, 5, 6)
    assert sm.data.dtype == np.float32
    assert sm.data[1, 2, 3] == 5.0


def test_squeezes_trailing_singleton_4d_dimension(make_nii):
    # A (x, y, z, 1) volume is unambiguously a single 3D map -> squeeze to 3D.
    data = np.zeros((3, 3, 3, 1), dtype=np.float32)
    data[1, 1, 1, 0] = 5.0
    path = make_nii(data, name="singleton.nii.gz")

    sm = load_stat_map(path)

    assert sm.data.shape == (3, 3, 3)
    assert sm.data[1, 1, 1] == 5.0


def test_raises_on_multi_volume_4d(make_nii):
    # Several volumes -> ambiguous which to show -> refuse rather than guess.
    data = np.zeros((3, 3, 3, 4), dtype=np.float32)
    path = make_nii(data, name="timeseries.nii.gz")

    with pytest.raises(ValueError, match=r"Expected a single 3D volume, got 4 volumes!"):
        load_stat_map(path)
