"""Atlas region labeling of stat-map voxels.

`extract_regions` is pure given a `RegionLabeler` — the protocol that abstracts
the atlas lookup, so tests inject a fake and never touch the network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from nilearn.image import resample_to_img
from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree

WHITE_MATTER = "Cerebral White Matter"


class RegionLabeler(Protocol):
    """Maps a voxel coordinate to an atlas region name."""

    def label_at(self, vox: tuple[int, int, int]) -> str: ...

    def nearest_cortical(self, vox: tuple[int, int, int]) -> str: ...


@dataclass
class AtlasLabeler:
    """A RegionLabeler backed by resampled Harvard-Oxford label arrays.

    Cortical labels take priority, subcortical is the fallback, and unlabeled
    voxels are "Unknown". The label arrays are injected (already resampled to the
    stat-map grid), so the lookup logic is testable without fetching an atlas.
    """

    cort: np.ndarray         # int label array, 0 = background
    cort_labels: list[str]   # label index -> name, [0] == "Background"
    sub: np.ndarray
    sub_labels: list[str]

    def label_at(self, vox: tuple[int, int, int]) -> str:
        i, j, k = vox
        ci = int(self.cort[i, j, k])
        if ci > 0:
            return self.cort_labels[ci]
        si = int(self.sub[i, j, k])
        if si > 0:
            return self.sub_labels[si]
        return "Unknown"

    def nearest_cortical(self, vox: tuple[int, int, int]) -> str:
        """Region of the cortical-labeled voxel closest to `vox`."""
        _, idx = self._cort_tree.query(vox)
        i, j, k = self._cort_points[idx]
        return self.cort_labels[int(self.cort[i, j, k])]

    @cached_property
    def _cort_points(self) -> np.ndarray:
        return np.argwhere(self.cort > 0)

    @cached_property
    def _cort_tree(self) -> cKDTree:
        return cKDTree(self._cort_points)


def strip_hemisphere(name: str) -> str:
    """Drop a leading 'Left '/'Right ' hemisphere prefix from an atlas label."""
    for prefix in ("Left ", "Right "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def extract_regions(mask: np.ndarray, labeler: RegionLabeler) -> dict[str, int]:
    """Count active (non-zero) voxels in `mask`, grouped by atlas label.

    Hemisphere is already encoded by which half of the volume is passed in, so a
    leading "Left "/"Right " prefix is stripped and counts merge by region.
    """
    counts: dict[str, int] = {}
    for vox in np.argwhere(mask != 0):
        name = strip_hemisphere(labeler.label_at(tuple(vox)))
        counts[name] = counts.get(name, 0) + 1
    return counts


def region_table(
    left: dict[str, int], right: dict[str, int]
) -> list[tuple[str, int, int]]:
    """Merge per-hemisphere region counts into (name, left, right) rows.

    Sorted by combined count (most active first); a region absent on one side
    gets 0 there, so one-sided activation is visible.
    """
    names = sorted(
        left.keys() | right.keys(),
        key=lambda n: -(left.get(n, 0) + right.get(n, 0)),
    )
    return [(n, left.get(n, 0), right.get(n, 0)) for n in names]


def region_totals(rows: list[tuple[str, int, int]]) -> tuple[int, int]:
    """Total (left, right) voxel counts across region rows."""
    return sum(r[1] for r in rows), sum(r[2] for r in rows)


def region_totals_ratio(left: int, right: int) -> tuple[float, float]:
    """L:R ratio normalized so the smaller side is 1.0."""
    smaller = min(left, right)
    return (left / smaller, right / smaller)


def laterality_index(
    rows: list[tuple[str, int, int]]
) -> list[tuple[str, float]]:
    """Per-region laterality index LI = (L - R) / (L + R).

    +1 is fully left-lateralized, -1 fully right, 0 symmetric. Input order is
    preserved. Regions reach this table only when present on at least one side,
    so L + R > 0 always holds.
    """
    return [(name, (left - right) / (left + right)) for name, left, right in rows]


def harvard_oxford_labeler(
    target_img, fetch_atlas: Callable[[str], object]
) -> AtlasLabeler:
    """Build an AtlasLabeler from Harvard-Oxford atlases resampled to `target_img`.

    `fetch_atlas(name)` returns an object with `.maps` (a label image) and
    `.labels` (list of names) — in production `nilearn`'s
    `fetch_atlas_harvard_oxford` (network), in tests a fake. Only the fetch
    touches the network; resampling is offline.
    """
    def _resampled(atlas) -> np.ndarray:
        img = resample_to_img(atlas.maps, target_img, interpolation="nearest")
        return img.get_fdata().astype(int)

    cort = fetch_atlas("cort-maxprob-thr25-2mm")
    sub = fetch_atlas("sub-maxprob-thr25-2mm")

    return AtlasLabeler(
        cort=_resampled(cort),
        cort_labels=cort.labels,
        sub=_resampled(sub),
        sub_labels=sub.labels,
    )
