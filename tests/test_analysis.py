"""Tests for hemistat.analysis geometry helpers.

Written first (RED): hemistat.analysis does not exist yet.

`vox_to_mni(affine, idx, axis)` converts a voxel slice index along one axis to
its MNI mm coordinate. It assumes a diagonal (axis-aligned) affine — the
invariant the pipeline guarantees by resampling to template space — so the
result is just `affine[axis, axis] * idx + affine[axis, 3]`, independent of the
off-axis voxel coords (and thus of volume shape).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hemistat.analysis import (
    active_slices,
    analyze_stat_map,
    calc_lateralization_score,
    mirror_pairs,
    reflect_across_midline,
    regions_by_hemisphere,
    save_json_results,
    split_hemispheres,
    vox_to_mni,
    wm_subregions,
)
from hemistat.io import StatMap
from hemistat.regions import AtlasLabeler
from tests.conftest import FakeLabeler


@pytest.fixture(autouse=True)
def _stub_atlas_labeler(monkeypatch):
    """analyze_stat_map builds a real Harvard-Oxford labeler; stub it so region
    extraction stays offline. Tests that assert on regions override this."""
    monkeypatch.setattr(
        "hemistat.analysis.harvard_oxford_labeler",
        lambda target, fetch_atlas: FakeLabeler({}),
    )


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


def test_analyze_includes_lateralization_score():
    # affine: mni_x = 2*i - 3  ->  x=1 mirror (x=2) is blank, so the
    # one active slice is fully lateralized -> score 1.0.
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

    assert analysis.lateralization_score == 1.0


def test_analyze_includes_sided_regions(monkeypatch):
    # affine: mni_x = 2*i - 3  ->  i=1 is left, i=3 is right.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 1.0   # left
    data[3, 0, 0] = 1.0   # right
    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=affine)

    # Override the autouse stub with a labeler that names these voxels.
    monkeypatch.setattr(
        "hemistat.analysis.harvard_oxford_labeler",
        lambda target, fetch_atlas: FakeLabeler(
            {(1, 0, 0): "Thalamus", (3, 0, 0): "Thalamus"}
        ),
    )

    analysis = analyze_stat_map(sm)

    assert analysis.sided_regions == [("Thalamus", 1, 1)]


def test_analyze_includes_wm_subregions(monkeypatch):
    # affine: mni_x = 2*i - 3  ->  i=1 is left.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cort = np.zeros((4, 1, 1), dtype=int)
    cort[0, 0, 0] = 1   # Precentral Gyrus
    sub = np.zeros((4, 1, 1), dtype=int)
    sub[1, 0, 0] = 1    # Left WM, nearest cortical = Precentral
    labeler = AtlasLabeler(
        cort=cort,
        cort_labels=["Background", "Precentral Gyrus"],
        sub=sub,
        sub_labels=["Background", "Left Cerebral White Matter"],
    )
    monkeypatch.setattr(
        "hemistat.analysis.harvard_oxford_labeler",
        lambda target, fetch_atlas: labeler,
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 1.0
    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=affine)

    analysis = analyze_stat_map(sm)

    assert analysis.wm_subregions == [("Precentral Gyrus", 1, 0)]


def test_wm_subregions_breaks_down_white_matter_by_nearest_cortical():
    # affine: mni_x = 2*i - 3  ->  i=0,1 left; i=2,3 right.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cort = np.zeros((4, 1, 2), dtype=int)
    cort[0, 0, 0] = 1   # Precentral Gyrus (left)
    cort[3, 0, 0] = 2   # Insular Cortex (right)
    sub = np.zeros((4, 1, 2), dtype=int)
    sub[1, 0, 0] = 1    # Left WM   (nearest cortical = Precentral)
    sub[1, 0, 1] = 1    # Left WM   (nearest cortical = Precentral)
    sub[2, 0, 0] = 2    # Right WM  (nearest cortical = Insular)
    labeler = AtlasLabeler(
        cort=cort,
        cort_labels=["Background", "Precentral Gyrus", "Insular Cortex"],
        sub=sub,
        # Real Harvard-Oxford WM labels are hemisphere-prefixed.
        sub_labels=["Background", "Left Cerebral White Matter", "Right Cerebral White Matter"],
    )
    data = np.zeros((4, 1, 2), dtype=np.float32)
    data[1, 0, 0] = 1.0
    data[1, 0, 1] = 1.0
    data[2, 0, 0] = 1.0
    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=affine)

    assert wm_subregions(sm, labeler) == [
        ("Precentral Gyrus", 2, 0),
        ("Insular Cortex", 0, 1),
    ]


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
    # affine: mni_x = 2*i - 3  ->  x=1's mirror (x=2) is blank.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0   # mirror blank -> all activation is unique

    assert calc_lateralization_score(data, affine) == 1.0


def test_lateralization_score_counts_off_grid_mirror_as_unique():
    # affine: mni_x = 2*i + 2  ->  x=1's mirror is off the grid (no counterpart).
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0   # mirror off-grid -> activation is unique

    assert calc_lateralization_score(data, affine) == 1.0


def test_lateralization_score_is_zero_when_no_activation():
    # Empty volume -> nothing to measure -> 0.0 (not NaN).
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)

    assert calc_lateralization_score(data, affine) == 0.0


def test_lateralization_score_is_partial_with_mix_of_unique_and_shared():
    # affine: mni_x = 2*i - 3  ->  mirrors 0<->3, 1<->2.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[0, 0, 0] = 4.0   # mirror (x=3) blank -> unique
    data[1, 0, 0] = 2.0   # shared mirror pair with x=2
    data[2, 0, 0] = 2.0

    # unique 4 / total 8 = 0.5
    assert calc_lateralization_score(data, affine) == 0.5


def test_lateralization_score_pools_voxels_globally():
    # affine: mni_x = 2*i - 3  ->  mirrors 1<->2; pooled over the whole volume.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 2), dtype=np.float32)
    data[1, 0, 0] = 1.0   # mirror (x=2 @ z0) active -> shared
    data[1, 0, 1] = 2.0   # mirror (x=2 @ z1) blank  -> unique
    data[2, 0, 0] = 5.0   # mirror (x=1 @ z0) active -> shared

    # unique 2 / total (1 + 2 + 5) = 2 / 8
    assert calc_lateralization_score(data, affine) == 0.25


def test_lateralization_score_matches_mirror_pairs_oracle():
    # An asymmetric volume with unique, shared, and edge (off-grid mirror) voxels.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -4.0],   # mni_x = 2*i - 4  ->  mirrors 0<->4, 1<->3, 2 is midline
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((5, 2, 2), dtype=np.float32)
    data[0, 1, 0] = 5.0   # left edge; mirror (i=4) blank -> unique
    data[1, 0, 0] = 3.0   # left; mirror (i=3 @ 0,0) blank -> unique
    data[1, 0, 1] = 2.0   # left; mirror (i=3 @ 0,1) active -> shared
    data[3, 0, 1] = 2.0   # right; mirror of (1, 0, 1)

    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=affine)

    assert (
        analyze_stat_map(sm).lateralization_score
        == _lateralization_via_mirror_pairs(data, affine)
    )


def _lateralization_via_mirror_pairs(data, affine):
    """The slice-pairing definition of the lateralization score.

    A frozen copy of the original mirror-pairs approach, kept as an independent
    cross-check. The production score will move to a whole-volume reflection;
    this proves the two definitions agree (grouping-invariance).
    """
    unique_total = grand_total = 0.0
    for idx, mirror_idx in mirror_pairs(data, affine, axis=0):
        stat_sl = np.take(data, idx, axis=0)
        mirror_sl = (
            np.take(data, mirror_idx, axis=0)
            if mirror_idx is not None
            else np.zeros_like(stat_sl)
        )
        unique_total += np.sum(np.abs(np.where(mirror_sl == 0, stat_sl, 0)))
        grand_total += np.sum(np.abs(stat_sl))
    return unique_total / grand_total if grand_total else 0.0

def test_reflect_across_midline_mirrors_volume_over_midline():
    # affine: mni_x = 2*i - 3  ->  mirrors 0<->3, 1<->2.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 5.0   # left of midline; its mirror is x=2

    mirrored = reflect_across_midline(data, affine)

    expected = np.zeros((4, 1, 1), dtype=np.float32)
    expected[2, 0, 0] = 5.0
    np.testing.assert_array_equal(mirrored, expected)


def test_regions_by_hemisphere_splits_labels_and_merges():
    # affine: mni_x = 2*i - 3  ->  i=0,1 are left (x<=0); i=2,3 are right.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 1.0   # left hemisphere
    data[3, 0, 0] = 1.0   # right hemisphere
    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=affine)
    labeler = FakeLabeler({(1, 0, 0): "Thalamus", (3, 0, 0): "Thalamus"})

    # One voxel each side, same region -> a single row with both columns set.
    assert regions_by_hemisphere(sm, labeler) == [("Thalamus", 1, 1)]


def test_save_json_results_writes_analysis_as_json(tmp_path, monkeypatch):
    # affine: mni_x = 2*i - 3  ->  i=1 is left, i=3 is right.
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros((4, 1, 1), dtype=np.float32)
    data[1, 0, 0] = 1.0   # left
    data[3, 0, 0] = 1.0   # right
    sm = StatMap(path=Path("t.nii.gz"), data=data, affine=affine)

    # Fake the labeler so region extraction stays offline and deterministic.
    monkeypatch.setattr(
        "hemistat.analysis.harvard_oxford_labeler",
        lambda target, fetch_atlas: FakeLabeler(
            {(1, 0, 0): "Thalamus", (3, 0, 0): "Thalamus"}
        ),
    )
    analysis = analyze_stat_map(sm)

    json_results_path = tmp_path / "results.json"
    save_json_results(analysis, json_results_path)

    written = json.loads(json_results_path.read_text())
    assert written == {
        "axial": [0],
        "coronal": [0],
        "sagittal": [1, 3],
        "mirror": [[1, 2], [3, 0]],
        "lateralization_score": 1.0,
        "sided_regions": [["Thalamus", 1, 1]],
        "wm_subregions": [],
    }
