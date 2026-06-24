"""Loading NIfTI stat maps into the `StatMap` data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class StatMap:
    """A statistical map loaded from a NIfTI file."""

    path: Path
    data: np.ndarray                  # float32 voxel data
    affine: np.ndarray | None = None  # 4x4 voxel-to-mm transform


def load_stat_map(path: str | Path) -> StatMap:
    """Load a NIfTI file into a StatMap."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return StatMap(path=Path(path), data=data, affine=img.affine)
