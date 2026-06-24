#!/usr/bin/env python3
"""Compute the L/R lateralization score for a NIfTI stat map.

Not part of the TDD suite — a thin CLI that composes the library functions:

    load_stat_map -> mirror_pairs -> lateralization_score

Usage:
    python scripts/lateralization_score.py <path_to_nii_file>
"""

import argparse

from hemistat.analysis import lateralization_score, mirror_pairs
from hemistat.io import load_stat_map


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nii_path", help="Path to a .nii or .nii.gz stat map")
    args = parser.parse_args()

    sm = load_stat_map(args.nii_path)
    pairs = mirror_pairs(sm.data, sm.affine, axis=0)
    score = lateralization_score(sm.data, pairs, axis=0)

    print(f"{sm.path.name}: lateralization score = {score:.3f}")


if __name__ == "__main__":
    main()
