"""Pure geometry/analysis helpers over stat-map voxel data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_harvard_oxford

from hemistat.io import StatMap
from hemistat.regions import (
    WHITE_MATTER,
    RegionLabeler,
    extract_regions,
    harvard_oxford_labeler,
    region_table,
    strip_hemisphere,
)


def reflect_across_midline(data: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Reflect the volume across the MNI x = 0 midline.

    Returns an array where each voxel holds the value of its left/right mirror
    (`out[x] == data[mirror_x]`); voxels whose mirror falls off the grid are
    zero-filled.
    """
    a, t = affine[0, 0], affine[0, 3]
    n = data.shape[0]
    out = np.zeros_like(data)
    for x in range(n):
        mirror_x = round((-vox_to_mni(affine, x, 0) - t) / a)
        if 0 <= mirror_x < n:
            out[x] = data[mirror_x]
    return out


def vox_to_mni(affine: np.ndarray, idx: int, axis: int) -> float:
    """MNI mm coordinate for a voxel slice index along the given axis.

    Assumes a diagonal (axis-aligned) affine.
    """
    return float(affine[axis, axis] * idx + affine[axis, 3])


def active_slices(data: np.ndarray, axis: int) -> list[int]:
    """Ascending indices of slices along `axis` that contain any non-zero voxel."""
    return [
        i for i in range(data.shape[axis])
        if np.count_nonzero(np.take(data, i, axis=axis)) > 0
    ]


def split_hemispheres(
    data: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split voxels into (left, right) hemispheres on MNI x (left is x <= 0)."""
    mni_xs = affine[0, 0] * np.arange(data.shape[0]) + affine[0, 3]
    is_left = (mni_xs <= 0)[:, np.newaxis, np.newaxis]
    return data * is_left, data * ~is_left


def regions_by_hemisphere(
    sm: StatMap, labeler: RegionLabeler
) -> list[tuple[str, int, int]]:
    """Per-hemisphere region counts: split on the midline, label each side, merge."""
    left, right = split_hemispheres(sm.data, sm.affine)
    return region_table(extract_regions(left, labeler), extract_regions(right, labeler))


def _wm_counts(mask: np.ndarray, labeler: RegionLabeler) -> dict[str, int]:
    """Count white-matter voxels in `mask` by their nearest cortical region."""
    counts: dict[str, int] = {}
    for vox in np.argwhere(mask != 0):
        v = tuple(vox)
        if strip_hemisphere(labeler.label_at(v)) == WHITE_MATTER:
            name = labeler.nearest_cortical(v)
            counts[name] = counts.get(name, 0) + 1
    return counts


def wm_subregions(
    sm: StatMap, labeler: RegionLabeler
) -> list[tuple[str, int, int]]:
    """Break white-matter voxels down by nearest cortical region, per hemisphere."""
    left, right = split_hemispheres(sm.data, sm.affine)
    return region_table(_wm_counts(left, labeler), _wm_counts(right, labeler))


def mirror_pairs(
    data: np.ndarray, affine: np.ndarray, axis: int = 0
) -> list[tuple[int, int | None]]:
    """Pair each active slice with its geometric mirror index across MNI x = 0.

    The mirror is purely geometric (from the affine), independent of whether the
    mirror slice contains activation; `None` means the mirror falls off the grid.
    """
    a = affine[axis, axis]
    t = affine[axis, 3]
    n = data.shape[axis]
    pairs = []
    for i in active_slices(data, axis):
        mirror_mni = -vox_to_mni(affine, i, axis)
        j = round((mirror_mni - t) / a)
        pairs.append((i, j if 0 <= j < n else None))
    return pairs


def calc_lateralization_score(data: np.ndarray, affine: np.ndarray) -> float:
    """Fraction of total activation whose left/right mirror voxel is blank.

    Reflects the whole volume across MNI x = 0 and divides the activation with a
    blank mirror by the total activation: a single global ratio over the whole
    volume, independent of slicing. 1.0 means fully lateralized; 0.0 when there
    is no activation.
    """
    mirrored = reflect_across_midline(data, affine)
    unique = np.where(mirrored == 0, data, 0)
    grand_total = np.sum(np.abs(data))
    return float(np.sum(np.abs(unique)) / grand_total) if grand_total > 0 else 0.0


@dataclass(frozen=True)
class StatMapAnalysis:
    """Results of analyzing a stat map, consumed by the renderer."""

    axial: list[int]                      # active slice indices, axis 2
    coronal: list[int]                    # active slice indices, axis 1
    sagittal: list[int]                   # active slice indices, axis 0
    mirror: list[tuple[int, int | None]]  # (slice, geometric mirror) on axis 0
    lateralization_score: float           # global share of activation unique to its side
    sided_regions: list[tuple[str, int, int]]    # (region, left, right)
    wm_subregions: list[tuple[str, int, int]]    # white matter broken down by nearest cortical


def save_json_results(analysis: StatMapAnalysis, json_results_path) -> None:
    """Serialize a StatMapAnalysis to JSON at json_results_path."""
    Path(json_results_path).write_text(json.dumps(asdict(analysis)))


def analyze_stat_map(sm: StatMap) -> StatMapAnalysis:
    """Run the analysis leaves over a stat map and collect them.

    Region labeling fetches the Harvard-Oxford atlas (cached after first use).
    """
    target = nib.Nifti1Image(sm.data, sm.affine)
    labeler = harvard_oxford_labeler(target, fetch_atlas=fetch_atlas_harvard_oxford)
    return StatMapAnalysis(
        axial=active_slices(sm.data, axis=2),
        coronal=active_slices(sm.data, axis=1),
        sagittal=active_slices(sm.data, axis=0),
        mirror=mirror_pairs(sm.data, sm.affine, axis=0),
        lateralization_score=calc_lateralization_score(sm.data, sm.affine),
        sided_regions=regions_by_hemisphere(sm, labeler),
        wm_subregions=wm_subregions(sm, labeler),
    )
