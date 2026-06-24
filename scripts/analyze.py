#!/usr/bin/env python3
"""Analyze a NIfTI stat map and print a summary.

Usage:
    python scripts/analyze.py <path_to_nii_file>
"""

import argparse

from hemistat.analysis import analyze_stat_map
from hemistat.io import load_stat_map


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


if __name__ == "__main__":
    main()
