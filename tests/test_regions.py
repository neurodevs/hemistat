"""Tests for hemistat.regions — atlas region labeling.

The atlas lookup is injected as a `RegionLabeler`, so these tests use a fake
backed by a small coordinate->name map and never fetch a real atlas.
"""

import types

import nibabel as nib
import numpy as np

from hemistat.regions import AtlasLabeler, extract_regions, harvard_oxford_labeler


class FakeLabeler:
    """A RegionLabeler backed by an explicit {(i, j, k): name} map."""

    def __init__(self, labels: dict[tuple[int, int, int], str]):
        self._labels = labels

    def label_at(self, vox: tuple[int, int, int]) -> str:
        return self._labels.get(tuple(vox), "Unknown")


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
