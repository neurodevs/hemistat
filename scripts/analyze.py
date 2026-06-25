#!/usr/bin/env python3
"""Analyze a NIfTI stat map and print a summary.

Usage:
    python scripts/analyze.py <path_to_nii_file>
"""

import argparse

from hemistat.analysis import analyze_stat_map
from hemistat.io import load_stat_map
from hemistat.regions import WHITE_MATTER, region_totals, region_totals_ratio


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
    total_left, total_right = region_totals(analysis.sided_regions)
    ratio_left, ratio_right = region_totals_ratio(total_left, total_right)
    print(
        f"    {total_left:>5} / {total_right:<5} TOTAL "
        f"(L {ratio_left:.1f} : R {ratio_right:.1f})"
    )
    for name, left, right in analysis.sided_regions:
        print(f"    {left:>5} / {right:<5} {name}")
        # White matter has no cortical label; show where it was re-mapped to.
        if name == WHITE_MATTER:
            for sub_name, sub_left, sub_right in analysis.wm_subregions:
                print(f"          ↳ {sub_left:>4} / {sub_right:<4} {sub_name} (WM)")


if __name__ == "__main__":
    main()
