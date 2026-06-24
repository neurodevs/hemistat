"""Pure geometry/analysis helpers over stat-map voxel data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hemistat.io import StatMap


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


@dataclass(frozen=True)
class StatMapAnalysis:
    """Results of analyzing a stat map, consumed by the renderer."""

    axial: list[int]      # active slice indices, axis 2
    coronal: list[int]    # active slice indices, axis 1
    sagittal: list[int]   # active slice indices, axis 0


def analyze_stat_map(sm: StatMap) -> StatMapAnalysis:
    """Run the analysis leaves over a stat map and collect them."""
    return StatMapAnalysis(
        axial=active_slices(sm.data, axis=2),
        coronal=active_slices(sm.data, axis=1),
        sagittal=active_slices(sm.data, axis=0),
    )
