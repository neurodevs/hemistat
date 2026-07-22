#!/usr/bin/env python3
"""Estimate regional brain volumes, left vs. right, from a directory of DICOMs.

Recurses into an arbitrary nesting of subdirectories, finds every DICOM file
regardless of name/extension, and reads pixel data for only the slices of
the single series that forms the primary anatomical volume.

The Harvard-Oxford atlas used for region labels is defined in MNI152
standard space, but a raw DICOM series is in the scanner's native subject
space (this subject's head size/position/orientation) — naively resampling
the atlas onto it misaligns every region. So before labeling, the DICOM
volume is registered to the MNI152 template with ANTs SyN (affine +
nonlinear deformable warp), which — unlike an affine-only fit — can bend to
match this subject's actual cortical folding, not just gross position and
size.

`load_mni152_template()` is NOT skull-stripped (~53% of its nonzero voxels
are skull/scalp/neck, not brain). Registering a skull-on native head to a
skull-on template lets SyN chase skull/scalp shape instead of locking onto
brain anatomy, which is worse than useless since skull shape varies a lot
person-to-person and carries no information about where the cortex actually
is. The fixed (template) side is masked to the intracranial volume via
`load_mni152_brain_mask()`, both by zeroing the template outside the mask
and by passing it as the registration metric mask, so SyN is only ever
scored on how well brain-interior structures line up. The moving (native)
side is left as-is — skull-stripping *that* would need real brain
extraction (e.g. antspynet's deep-learning model), a much heavier
dependency left out here for now.

This registration step is intentionally kept local to this script rather
than added to hemistat's library code, since it's still a rough fit
(default ANTs settings, no quality-controlled pipeline) good enough for a
volume estimate, not the accuracy a clinical/research pipeline would demand.

Usage:
    python scripts/dicom_regional_volumes.py <dicom_directory>
"""

import argparse

import ants
import nibabel as nib
import numpy as np
from nilearn.datasets import (
    fetch_atlas_harvard_oxford,
    load_mni152_brain_mask,
    load_mni152_template,
)

from hemistat.analysis import regions_by_hemisphere
from hemistat.dicom_io import load_dicom_directory
from hemistat.io import StatMap
from hemistat.regions import (
    harvard_oxford_labeler,
    region_totals,
    region_totals_ratio,
)


def _to_ants(data: np.ndarray, affine: np.ndarray) -> ants.ANTsImage:
    """Convert a canonical-RAS (data, affine) pair to an ANTsImage (LPS).

    Assumes `affine` is diagonal with a positive diagonal, which is what
    `nib.as_closest_canonical` produces — array indices increase along
    +R/+A/+S. ANTs (like DICOM) uses LPS, so origin/direction flip x/y
    relative to RAS.
    """
    return ants.from_numpy(
        data.astype(np.float32),
        origin=(-float(affine[0, 3]), -float(affine[1, 3]), float(affine[2, 3])),
        spacing=tuple(float(v) for v in np.diagonal(affine[:3, :3])),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )


def register_to_mni(sm: StatMap, template_resolution: int = 2) -> StatMap:
    """Register `sm` onto the MNI152 template grid with ANTs SyN.

    Rigid + affine + deformable warp, so atlas-based region labels line up
    with this subject's actual anatomy instead of being resampled onto raw
    scanner-space coordinates or a coarse linear approximation of it.
    """
    template = nib.as_closest_canonical(load_mni152_template(resolution=template_resolution))
    brain_mask = nib.as_closest_canonical(load_mni152_brain_mask(resolution=template_resolution))
    template_data = template.get_fdata().astype(np.float32) * (brain_mask.get_fdata() > 0)

    fixed = _to_ants(template_data, template.affine)
    fixed_mask = _to_ants((brain_mask.get_fdata() > 0).astype(np.float32), brain_mask.affine)
    moving = _to_ants(sm.data, sm.affine)

    result = ants.registration(
        fixed=fixed, moving=moving, type_of_transform="SyNRA", mask=fixed_mask
    )
    # `mask=` only restricts what the registration *metric* is scored on; the
    # warped output still carries the native volume's full content (skull,
    # face, neck) resampled into template space. Zero it to intracranial
    # voxels here too, or all that extra tissue floods into "Unknown" and
    # inflates every volume total.
    warped = result["warpedmovout"].numpy() * (brain_mask.get_fdata() > 0)
    return StatMap(
        path=sm.path,
        data=warped.astype(np.float32),
        affine=template.affine,
    )


def voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.prod(np.diagonal(affine[:3, :3]))))


# Rough adult full-head extents (mm) below which a scan is almost certainly
# a partial-coverage acquisition, not a whole-head volume. No amount of
# registration can recover anatomy that was never scanned, so this is
# checked before trusting any regional breakdown.
MIN_PLAUSIBLE_EXTENT_MM = {"R/L (width)": 120.0, "A/P (depth)": 140.0, "S/I (height)": 140.0}


def check_coverage(sm: StatMap) -> None:
    """Warn if the native DICOM volume looks too small to be a full head."""
    extents = np.abs(np.diagonal(sm.affine[:3, :3])) * np.array(sm.data.shape)
    print("  physical extent (mm):")
    undersized = False
    for (label, min_mm), extent in zip(MIN_PLAUSIBLE_EXTENT_MM.items(), extents):
        flag = " <-- SHORT" if extent < min_mm else ""
        print(f"    {label}: {extent:.1f} mm (need >~{min_mm:.0f} mm for full coverage){flag}")
        undersized = undersized or extent < min_mm
    if undersized:
        print(
            "  WARNING: this looks like a partial-coverage acquisition, not a "
            "whole head. Region volumes below will be unreliable regardless of "
            "registration quality — check that the right DICOM series was selected."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dicom_dir", help="Directory containing DICOM files (any nesting)")
    args = parser.parse_args()

    native = load_dicom_directory(args.dicom_dir)
    print("\n", native.path)
    print(f"  native grid: {native.data.shape}")
    check_coverage(native)
    print("  registering to MNI152...")
    sm = register_to_mni(native)
    vox_vol = voxel_volume_mm3(sm.affine)

    target = nib.Nifti1Image(sm.data, sm.affine)
    labeler = harvard_oxford_labeler(target, fetch_atlas=fetch_atlas_harvard_oxford)
    rows = regions_by_hemisphere(sm, labeler)

    print(f"  voxel volume: {vox_vol:.2f} mm^3   grid: {sm.data.shape}")

    total_left, total_right = region_totals(rows)
    ratio_left, ratio_right = region_totals_ratio(total_left, total_right)
    side = "L" if ratio_left >= ratio_right else "R"
    dominant = max(ratio_left, ratio_right)
    print("  regions (L / R volume in cm^3):")
    print(
        f"    {total_left * vox_vol / 1000:>9.1f} / {total_right * vox_vol / 1000:<9.1f}"
        f"   TOTAL  ({side} {dominant:.2f} : 1)"
    )
    for name, left, right in rows:
        print(
            f"    {left * vox_vol / 1000:>9.1f} / {right * vox_vol / 1000:<9.1f}"
            f"   {name}"
        )


if __name__ == "__main__":
    main()
