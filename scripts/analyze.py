#!/usr/bin/env python3
"""Analyze a NIfTI stat map and print a summary.

Usage:
    python scripts/analyze.py <path_to_nii_file>
"""

import argparse

from pathlib import Path

from hemistat.analysis import analyze_stat_map, save_json_results
from hemistat.io import load_stat_map
from hemistat.regions import WHITE_MATTER, region_totals, region_totals_ratio


def side_letter(li: float) -> str:
    """L/R tag for a laterality index (right-positive); blank when symmetric."""
    if li > 0:
        return "R"
    if li < 0:
        return "L"
    return " "


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nii_path", help="Path to a .nii or .nii.gz stat map")
    args = parser.parse_args()

    sm = load_stat_map(args.nii_path)
    analysis = analyze_stat_map(sm)
    stem = Path(args.nii_path).name.removesuffix(".gz").removesuffix(".nii")
    save_json_results(analysis, f"./artifacts/{stem}.json")

    print("\n", sm.path.name)
    print(
        f"  active slices:  axial={len(analysis.axial)} "
        f"coronal={len(analysis.coronal)} sagittal={len(analysis.sagittal)}"
    )
    print(f"  mirror pairs:   {len(analysis.mirror)}")
    print(f"  lateralization: {analysis.lateralization_score:.3f}")
    print("  regions (L / R, LI: +1 right .. -1 left):")
    total_left, total_right = region_totals(analysis.sided_regions)
    ratio_left, ratio_right = region_totals_ratio(total_left, total_right)
    side = "L" if ratio_left >= ratio_right else "R"
    dominant = max(ratio_left, ratio_right)
    print(
        f"    {total_left:>5} / {total_right:<5}        TOTAL  ({side} {dominant:.2f} : 1)"
    )
    li_by_name = dict(analysis.region_laterality)
    wm_li_by_name = dict(analysis.wm_laterality)
    for name, left, right in analysis.sided_regions:
        li = li_by_name[name]
        print(f"    {left:>5} / {right:<5} {li:>+5.2f} {side_letter(li)}  {name}")
        # White matter has no cortical label; show where it was re-mapped to.
        if name == WHITE_MATTER:
            for sub_name, sub_left, sub_right in analysis.wm_subregions:
                sub_li = wm_li_by_name[sub_name]
                print(
                    f"          ↳ {sub_left:>4} / {sub_right:<4} "
                    f"{sub_li:>+5.2f} {side_letter(sub_li)}  {sub_name} (WM)"
                )


if __name__ == "__main__":
    main()
