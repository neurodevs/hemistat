"""Loading a DICOM series (found anywhere under a directory tree) into the
`StatMap` data contract, so it flows through the same regional/hemisphere
analysis already built for NIfTI stat maps.

DICOM exports commonly nest files arbitrarily deep and ship with no
extension (sometimes even an executable bit), so files are found by content,
not name. Only the slices belonging to the one series chosen as the primary
anatomical volume are ever read with pixel data; everything else is either
skipped or header-scanned then discarded.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset

from hemistat.io import StatMap, _require_diagonal_affine


def _scan_headers(root: str | Path) -> dict[str, list[Path]]:
    """Recursively read DICOM headers under `root`, grouped by series.

    A file counts as DICOM only if it parses as one; extension and
    permission bits are ignored. Pixel data is not read at this stage.
    """
    groups: dict[str, list[Path]] = {}
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        if not hasattr(ds, "SeriesInstanceUID"):
            continue
        groups.setdefault(str(ds.SeriesInstanceUID), []).append(path)
    return groups


def _is_volume_series(path: Path) -> bool:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return (
        hasattr(ds, "ImagePositionPatient")
        and hasattr(ds, "ImageOrientationPatient")
        and hasattr(ds, "PixelSpacing")
    )


def _slice_normal(iop: np.ndarray) -> np.ndarray:
    row_cosine, col_cosine = iop[:3], iop[3:]
    return np.cross(row_cosine, col_cosine)


# Rough adult full-head extents (mm), in patient-space (L/R, A/P, S/I) order.
# A series short of these on any axis is a partial/thin-slab acquisition
# (e.g. a targeted high-res sequence), not a whole head, regardless of how
# many slices it has.
_MIN_HEAD_EXTENT_MM = np.array([120.0, 140.0, 140.0])


def _series_extents_mm(paths: list[Path]) -> np.ndarray:
    """Approximate world-axis-aligned bounding box extents (L/R, A/P, S/I).

    Header-only (no pixel data read). Treats the oriented image-plus-slice-
    stack box as an axis-aligned bounding box in DICOM's LPS patient space —
    exact for on-axis axial/sagittal/coronal acquisitions, a reasonable
    approximation for anything oblique.
    """
    datasets = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in paths]
    iop = np.array(datasets[0].ImageOrientationPatient, dtype=float)
    row_cosine, col_cosine = iop[:3], iop[3:]
    normal = _slice_normal(iop)

    row_spacing, col_spacing = (float(v) for v in datasets[0].PixelSpacing)
    width = float(datasets[0].Columns) * col_spacing  # along row_cosine
    height = float(datasets[0].Rows) * row_spacing  # along col_cosine

    projections = [
        float(np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal))
        for ds in datasets
    ]
    depth = max(projections) - min(projections)  # along normal

    return (
        np.abs(row_cosine) * width
        + np.abs(col_cosine) * height
        + np.abs(normal) * depth
    )


def select_primary_series(groups: dict[str, list[Path]]) -> list[Path]:
    """Pick the series most likely to be the intended anatomical volume.

    A candidate needs geometry tags to be reconstructable into a 3D volume
    and more than one slice, which rules out scouts/localizers. Slice count
    alone isn't a safe tiebreaker beyond that, though: a thin, high-res,
    partial-coverage sequence can easily out-count a full-head series with
    fewer, thicker slices. So candidates are first restricted to ones whose
    physical extent actually covers a whole head (regardless of whether they
    were acquired axially, sagittally, or coronally); the most-sliced series
    among those wins.
    """
    candidates = [
        paths for paths in groups.values()
        if len(paths) > 1 and _is_volume_series(paths[0])
    ]
    if not candidates:
        raise ValueError(
            "No DICOM series with enough slices for a 3D volume was found!"
        )

    full_head = [
        paths for paths in candidates
        if np.all(_series_extents_mm(paths) >= _MIN_HEAD_EXTENT_MM)
    ]
    if not full_head:
        raise ValueError(
            "No DICOM series covers a full head (need >=120mm R/L, >=140mm "
            "A/P, >=140mm S/I) — every candidate looks like a partial or "
            "thin-slab acquisition."
        )
    return max(full_head, key=len)


def _load_and_sort_slices(paths: list[Path]) -> list[FileDataset]:
    datasets = [pydicom.dcmread(str(p)) for p in paths]
    normal = _slice_normal(
        np.array(datasets[0].ImageOrientationPatient, dtype=float)
    )
    datasets.sort(
        key=lambda ds: float(
            np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal)
        )
    )
    return datasets


def _build_affine(datasets: list[FileDataset]) -> np.ndarray:
    """Build a RAS-convention voxel-to-mm affine from sorted slice geometry.

    DICOM's patient coordinate system is LPS; the first two rows are negated
    to convert to the RAS convention nibabel/nilearn expect (+x = Right),
    the same conversion dcm2niix performs.
    """
    iop = np.array(datasets[0].ImageOrientationPatient, dtype=float)
    row_cosine, col_cosine = iop[:3], iop[3:]
    normal = _slice_normal(iop)

    row_spacing, col_spacing = (float(v) for v in datasets[0].PixelSpacing)
    ipp0 = np.array(datasets[0].ImagePositionPatient, dtype=float)

    if len(datasets) > 1:
        ipp1 = np.array(datasets[1].ImagePositionPatient, dtype=float)
        slice_spacing = float(np.dot(ipp1 - ipp0, normal))
    else:
        slice_spacing = float(getattr(datasets[0], "SpacingBetweenSlices", 1.0))

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_cosine * col_spacing
    affine_lps[:3, 1] = col_cosine * row_spacing
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = ipp0

    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps


def _slice_pixels(ds: FileDataset) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return ds.pixel_array.astype(np.float32).T * slope + intercept


def load_dicom_series(paths: list[Path]) -> StatMap:
    """Load one already-selected DICOM series into a StatMap, RAS-canonical."""
    datasets = _load_and_sort_slices(paths)
    volume = np.stack([_slice_pixels(ds) for ds in datasets], axis=2)
    affine = _build_affine(datasets)

    img = nib.as_closest_canonical(nib.Nifti1Image(volume, affine))
    _require_diagonal_affine(img.affine)
    return StatMap(
        path=Path(paths[0]).parent,
        data=img.get_fdata().astype(np.float32),
        affine=img.affine,
    )


def load_dicom_directory(root: str | Path) -> StatMap:
    """Find, select, and load the primary anatomical DICOM series under `root`.

    Recurses into an arbitrary nesting of subdirectories; only the slices of
    the chosen series are ever read with pixel data.
    """
    groups = _scan_headers(root)
    if not groups:
        raise ValueError(f"No DICOM files found under {root}!")
    primary = select_primary_series(groups)
    return load_dicom_series(primary)
