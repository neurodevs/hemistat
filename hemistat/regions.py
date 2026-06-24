"""Atlas region labeling of stat-map voxels.

`extract_regions` is pure given a `RegionLabeler` — the protocol that abstracts
the atlas lookup, so tests inject a fake and never touch the network.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RegionLabeler(Protocol):
    """Maps a voxel coordinate to an atlas region name."""

    def label_at(self, vox: tuple[int, int, int]) -> str: ...


def extract_regions(mask: np.ndarray, labeler: RegionLabeler) -> dict[str, int]:
    """Count active (non-zero) voxels in `mask`, grouped by atlas label."""
    counts: dict[str, int] = {}
    for vox in np.argwhere(mask != 0):
        name = labeler.label_at(tuple(vox))
        counts[name] = counts.get(name, 0) + 1
    return counts
