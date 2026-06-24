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
    data: np.ndarray    # float32 voxel data
    affine: np.ndarray  # 4x4 voxel-to-mm transform


def _to_single_volume(data: np.ndarray) -> np.ndarray:
    """Reduce a NIfTI array to a single 3D volume.

    A trailing singleton dimension (x, y, z, 1) is squeezed away; it raises on 
    ambiguous multi-volume 4D arrays rather than guessing which to use.
    """
    if data.ndim != 4:
        return data
    if data.shape[3] == 1:
        return data[..., 0]
    raise ValueError(f"Expected a single 3D volume, got {data.shape[3]} volumes!")


def _require_diagonal_affine(affine: np.ndarray) -> None:
    """Reject an oblique affine.

    Canonicalization fixes axis permutations and flips, but not genuine rotation;
    the geometry helpers assume an axis-aligned (diagonal) affine.
    """
    rotation = affine[:3, :3]
    off_diagonal = rotation - np.diag(np.diagonal(rotation))
    if not np.allclose(off_diagonal, 0.0, atol=1e-4):
        raise ValueError(
            f"Expected an axis-aligned affine; got an oblique one:\n{rotation}"
        )


def load_stat_map(path: str | Path) -> StatMap:
    """Load a NIfTI file into a StatMap."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    _require_diagonal_affine(img.affine)
    data = _to_single_volume(img.get_fdata().astype(np.float32))
    return StatMap(path=Path(path), data=data, affine=img.affine)
