"""Tests for hemistat.regions — atlas region labeling.

The atlas lookup is injected as a `RegionLabeler`, so these tests use a fake
backed by a small coordinate->name map and never fetch a real atlas.
"""

import types

import nibabel as nib
import numpy as np

from hemistat.regions import (
    AtlasLabeler,
    extract_regions,
    harvard_oxford_labeler,
    region_table,
    region_totals,
    region_totals_ratio,
)
from tests.conftest import FakeLabeler


def test_extract_regions_counts_active_voxels_by_label():
    data = np.zeros((3, 3, 3), dtype=np.float32)
    data[0, 0, 0] = 1.0
    data[1, 0, 0] = 1.0
    data[2, 0, 0] = 1.0
    
    labeler = FakeLabeler(
        {
            (0, 0, 0): "Precentral Gyrus",
            (1, 0, 0): "Precentral Gyrus",
            (2, 0, 0): "Insular Cortex",
        }
    )

    assert extract_regions(data, labeler) == {
        "Precentral Gyrus": 2,
        "Insular Cortex": 1,
    }


def test_extract_regions_strips_hemisphere_prefixes():
    data = np.zeros((3, 3, 3), dtype=np.float32)
    data[0, 0, 0] = 1.0
    data[1, 0, 0] = 1.0

    labeler = FakeLabeler(
        {
            (0, 0, 0): "Left Thalamus",
            (1, 0, 0): "Right Thalamus",
        }
    )

    # Hemisphere is tracked by which half is passed in, so the L/R prefix is
    # dropped and both voxels merge under the bare region name.
    assert extract_regions(data, labeler) == {"Thalamus": 2}


def test_atlas_labeler_returns_cortical_label():
    cort = np.zeros((2, 2, 2), dtype=int)
    cort[0, 0, 0] = 1
    sub = np.zeros((2, 2, 2), dtype=int)

    labeler = AtlasLabeler(
        cort=cort,
        cort_labels=["Background", "Precentral Gyrus"],
        sub=sub,
        sub_labels=["Background"],
    )

    assert labeler.label_at((0, 0, 0)) == "Precentral Gyrus"


def test_atlas_labeler_falls_back_to_subcortical_label():
    cort = np.zeros((2, 2, 2), dtype=int)   # no cortical label here
    sub = np.zeros((2, 2, 2), dtype=int)
    sub[0, 0, 0] = 1

    labeler = AtlasLabeler(
        cort=cort,
        cort_labels=["Background"],
        sub=sub,
        sub_labels=["Background", "Thalamus"],
    )

    assert labeler.label_at((0, 0, 0)) == "Thalamus"


def test_atlas_labeler_returns_unknown_when_unlabeled():
    cort = np.zeros((2, 2, 2), dtype=int)
    sub = np.zeros((2, 2, 2), dtype=int)

    labeler = AtlasLabeler(
        cort=cort,
        cort_labels=["Background"],
        sub=sub,
        sub_labels=["Background"],
    )

    assert labeler.label_at((0, 0, 0)) == "Unknown"


def test_atlas_labeler_prefers_cortical_over_subcortical():
    cort = np.zeros((2, 2, 2), dtype=int)
    cort[0, 0, 0] = 1
    sub = np.zeros((2, 2, 2), dtype=int)
    sub[0, 0, 0] = 1   # both labeled at the same voxel
    
    labeler = AtlasLabeler(
        cort=cort,
        cort_labels=["Background", "Precentral Gyrus"],
        sub=sub,
        sub_labels=["Background", "Thalamus"],
    )

    assert labeler.label_at((0, 0, 0)) == "Precentral Gyrus"


def test_atlas_labeler_keeps_white_matter_but_exposes_nearest_cortical():
    cort = np.zeros((3, 1, 1), dtype=int)
    cort[0, 0, 0] = 1   # Precentral Gyrus at x=0
    sub = np.zeros((3, 1, 1), dtype=int)
    sub[1, 0, 0] = 1    # Cerebral White Matter at x=1 (no cortical label there)

    labeler = AtlasLabeler(
        cort=cort,
        cort_labels=["Background", "Precentral Gyrus"],
        sub=sub,
        sub_labels=["Background", "Cerebral White Matter"],
    )

    # WM stays its own label; nearest_cortical exposes the closest cortical region.
    assert labeler.label_at((1, 0, 0)) == "Cerebral White Matter"
    assert labeler.nearest_cortical((1, 0, 0)) == "Precentral Gyrus"


def test_harvard_oxford_labeler_builds_from_fetched_atlases():
    # Fake fetch returns synthetic atlases on the target grid -> no network.
    affine = np.eye(4)
    target = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine)

    cort_map = np.zeros((2, 2, 2), dtype=np.int16)
    cort_map[0, 0, 0] = 1
    sub_map = np.zeros((2, 2, 2), dtype=np.int16)
    sub_map[1, 0, 0] = 1

    def fake_fetch(name):
        if name.startswith("cort"):
            return types.SimpleNamespace(
                maps=nib.Nifti1Image(cort_map, affine),
                labels=["Background", "Precentral Gyrus"],
            )
        return types.SimpleNamespace(
            maps=nib.Nifti1Image(sub_map, affine),
            labels=["Background", "Thalamus"],
        )

    labeler = harvard_oxford_labeler(target, fetch_atlas=fake_fetch)

    # Both atlases wired through: cortical voxel and subcortical voxel resolve.
    resolved = (labeler.label_at((0, 0, 0)), labeler.label_at((1, 0, 0)))
    assert resolved == ("Precentral Gyrus", "Thalamus")


def test_region_table_merges_left_and_right_counts():
    left = {"Thalamus": 50, "Insular Cortex": 10}
    right = {"Thalamus": 12, "Precentral Gyrus": 30}

    # (name, left, right), sorted by combined count; absent side -> 0.
    assert region_table(left, right) == [
        ("Thalamus", 50, 12),         # combined 62
        ("Precentral Gyrus", 0, 30),  # combined 30, left-absent
        ("Insular Cortex", 10, 0),    # combined 10, right-absent
    ]


def test_region_totals_sums_each_side():
    rows = [("Thalamus", 50, 12), ("Insular Cortex", 10, 0)]

    assert region_totals(rows) == (60, 12)


def test_region_totals_ratio_normalizes_smaller_side_to_one():
    # 448 : 256  ->  divide by the smaller side so it becomes 1.0.
    assert region_totals_ratio(448, 256) == (1.75, 1.0)
