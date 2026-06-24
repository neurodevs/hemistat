"""Tests for hemistat.regions — atlas region labeling.

The atlas lookup is injected as a `RegionLabeler`, so these tests use a fake
backed by a small coordinate->name map and never fetch a real atlas.
"""

import numpy as np

from hemistat.regions import extract_regions


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
