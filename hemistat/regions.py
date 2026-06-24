"""Atlas region labeling of stat-map voxels.

`extract_regions` is pure given a `RegionLabeler` — the protocol that abstracts
the atlas lookup, so tests inject a fake and never touch the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RegionLabeler(Protocol):
    """Maps a voxel coordinate to an atlas region name."""

    def label_at(self, vox: tuple[int, int, int]) -> str: ...


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


def extract_regions(mask: np.ndarray, labeler: RegionLabeler) -> dict[str, int]:
    """Count active (non-zero) voxels in `mask`, grouped by atlas label."""
    counts: dict[str, int] = {}
    for vox in np.argwhere(mask != 0):
        name = labeler.label_at(tuple(vox))
        counts[name] = counts.get(name, 0) + 1
    return counts
