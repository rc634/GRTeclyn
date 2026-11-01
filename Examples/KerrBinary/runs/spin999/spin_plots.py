#!/usr/bin/env python3
"""spin999 study: horizon histories of several runs in one figure.

    python spin_plots.py -o out.png --hole 1 RUN:label RUN:label ...

Panels: coordinate radius r_mean, chi_akv - chi_akv(0), M_irr / M_irr(0) - 1 (all against t).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spin_report import horizons

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc4", "#5a5a5a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--output", default="spin_plots.png")
    ap.add_argument("--hole", type=int, default=1)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, spec in enumerate(args.runs):
        path, _, label = spec.partition(":")
        h = horizons(Path(path)).get(args.hole)
        if h is None or not len(h["t"]):
            continue
        c = COLOURS[k % len(COLOURS)]
        t = h["t"]
        axes[0].plot(t, h["r_mean"], color=c, lw=1.6, label=label or Path(path).name)
        axes[1].plot(t, h["chi_akv"] - h["chi_akv"][0], color=c, lw=1.6)
        axes[2].plot(t, h["M_A"] / h["M_A"][0] - 1, color=c, lw=1.6)
    axes[0].set_ylabel("coordinate radius r_mean")
    axes[1].set_ylabel(r"$\chi_{\rm AKV}(t) - \chi_{\rm AKV}(0)$")
    axes[2].set_ylabel(r"$M_{\rm irr}(t) / M_{\rm irr}(0) - 1$")
    for ax in axes:
        ax.set_xlabel("t / M")
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].axhline(0, color="#999999", lw=0.8)
    axes[2].axhline(0, color="#999999", lw=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
