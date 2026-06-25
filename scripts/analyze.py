#!/usr/bin/env python3
"""Analyze a NIfTI stat map and print a summary.

Usage:
    python scripts/analyze.py <path_to_nii_file>
"""

import argparse

from hemistat.analysis import analyze_stat_map
from hemistat.io import load_stat_map
from hemistat.regions import WHITE_MATTER


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nii_path", help="Path to a .nii or .nii.gz stat map")
    args = parser.parse_args()

    sm = load_stat_map(args.nii_path)
    analysis = analyze_stat_map(sm)

    print("\n", sm.path.name)
    print(
        f"  active slices:  axial={len(analysis.axial)} "
        f"coronal={len(analysis.coronal)} sagittal={len(analysis.sagittal)}"
    )
    print(f"  mirror pairs:   {len(analysis.mirror)}")
    print(f"  lateralization: {analysis.lateralization_score:.3f}")
    print("  regions (L / R):")
    for name, left, right in analysis.sided_regions:
        print(f"    {left:>5} / {right:<5} {name}")
        # White matter has no cortical label; show where it was re-mapped to.
        if name == WHITE_MATTER:
            for sub_name, sub_left, sub_right in analysis.wm_subregions:
                print(f"            ↳ {sub_left:>5} / {sub_right:<5} {sub_name} (WM)")


if __name__ == "__main__":
    main()
