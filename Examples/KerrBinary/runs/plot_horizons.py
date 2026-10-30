#!/usr/bin/env python3
"""Plot the SAHF horizon data of a run against t: the Christodoulou mass M,
the momentum |P| and the dimensionless spin chi = sign(S_z) |S| / M^2, with
the spin from approximate Killing vectors (AKV, solid) and from coordinate
rotations (Komar-type integral, dashed).

    cd runs/flyby_sahf
    ~/.venvs/bibs-vis/bin/python ../plot_horizons.py              # show
    ~/.venvs/bibs-vis/bin/python ../plot_horizons.py --save h.png # save

Reads sahf_output/horizon_<id>.dat; gaps in a line are steps where that
horizon was not found.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLOURS = ("#1b7f79", "#c2410c", "#4c4cc2")  # H1, H2, H3 (H1/H2 checked for colour blindness)


def read_horizon(path):
    """{column: array} from a SmallDataIO file (first '#' line names the
    columns; skips a half-written last line)."""
    names, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            names = names or line[1:].split()
            continue
        parts = line.split()
        if names and len(parts) == len(names):
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                pass
    table = np.array(rows).reshape(-1, len(names or [0]))
    return dict(zip(names or [], table.T))


def gaps_to_nan(t, y):
    """Break the line where the finder missed steps."""
    if t.size < 3:
        return t, y
    cut = np.where(np.diff(t) > 1.5 * np.median(np.diff(t)))[0] + 1
    return np.insert(t, cut, np.nan), np.insert(y, cut, np.nan)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", default=".", type=Path)
    parser.add_argument("--save", metavar="FILE", help="write the figure instead of showing it")
    args = parser.parse_args()

    paths = sorted((args.run_dir / "sahf_output").glob("horizon_*.dat"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise SystemExit(f"no horizon files in {(args.run_dir / 'sahf_output').resolve()}")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.5, 7.5),
                             layout="constrained")
    horizon_handles = []
    for path in paths:
        hid = int(path.stem.split("_")[1])
        h = read_horizon(path)
        if not h.get("time", np.empty(0)).size:
            continue
        t = h["time"]
        P = np.sqrt(h["Px"] ** 2 + h["Py"] ** 2 + h["Pz"] ** 2)
        M, chi = {}, {}
        for kind in ("akv", "rot"):
            S = [h[f"S{c}_{kind}"] for c in "xyz"]
            M[kind] = h[f"M_T_{kind}"]
            chi[kind] = np.sign(S[2]) * np.sqrt(sum(s**2 for s in S)) / M[kind] ** 2
        colour = COLOURS[(hid - 1) % len(COLOURS)]
        axes[1].plot(*gaps_to_nan(t, P), color=colour, lw=1.5)
        for kind, style in (("rot", "--"), ("akv", "-")):
            axes[0].plot(*gaps_to_nan(t, M[kind]), color=colour, lw=1.5, ls=style)
            axes[2].plot(*gaps_to_nan(t, chi[kind]), color=colour, lw=1.5, ls=style)
        horizon_handles.append(Line2D(
            [], [], color=colour, lw=2,
            label=f"H{hid}  t = {t[-1]:.2f}:  M {M['akv'][-1]:.5f} (Komar {M['rot'][-1]:.5f})"
                  f"   |P| {P[-1]:.4f}   chi {chi['akv'][-1]:+.4f} (Komar {chi['rot'][-1]:+.4f})"))

    axes[0].set_ylabel(r"$M$ (Christodoulou)")
    axes[1].set_ylabel(r"$|P|$")
    axes[2].set_ylabel(r"$\chi = \mathrm{sign}(S_z)\,|S|/M^2$")
    axes[2].axhline(0.0, color="0.6", lw=0.8)
    axes[2].set_xlabel(r"$t$")
    for ax in axes:
        ax.grid(color="0.92")
    axes[0].legend(handles=horizon_handles, frameon=False, fontsize=7.5)
    axes[2].legend(handles=[Line2D([], [], color="0.3", lw=1.5, label="AKV"),
                            Line2D([], [], color="0.3", lw=1.5, ls="--",
                                   label="Komar (coordinate rotations)")],
                   frameon=False, fontsize=8)
    fig.suptitle(f"SAHF horizons    {args.run_dir.resolve().name}")

    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
