"""Collect ``metrics.json`` files under a results folder into a Markdown table next to the paper numbers.

Example:
    python tools/summarize.py results
"""
import glob
import json
import os
import sys

# Table 3 of the paper: OA, Kappa, F1, time (s).
PAPER = {
    "hetdata1": (0.980, 0.910, 0.921, 73.96),
    "hetdata2": (0.989, 0.840, 0.845, 12.19),
    "hetdata3": (0.985, 0.878, 0.885, 80.01),
    "hetdata4": (0.979, 0.807, 0.818, 13.19),
    "hetdata5": (0.931, 0.737, 0.778, 26.69),
    "hetdata6": (0.987, 0.843, 0.850, 77.80),
}


def main(root: str = "results"):
    print("| Dataset | OA | Kappa | F1 | Time (s) | Paper OA / Kappa / F1 | ΔF1 |")
    print("|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        p = PAPER.get(m["dataset"])
        ref = f"{p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f}" if p else "-"
        delta = f"{m['F1'] - p[2]:+.3f}" if p else "-"
        print(f"| {m['dataset']} | {m['OA']:.3f} | {m['Kappa']:.3f} | {m['F1']:.3f} | {m['time_s']:.1f} | {ref} | {delta} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
