#!/usr/bin/env python3
"""Watch a running KerrBinary simulation: the newest plotfile, refreshed to
each new one as it appears.

    chi      K          slices through the holes' plane (z of the punctures),
    |Ham|    |Psi4|     refinement levels outlined, SAHF horizons drawn

    cd runs/flyby_sahf
    ~/.venvs/bibs-vis/bin/python ../watch-live.py                 # both holes
    ~/.venvs/bibs-vis/bin/python ../watch-live.py --centre 2 --window 1.5
    ~/.venvs/bibs-vis/bin/python ../watch-live.py --window 100    # the waves

Colour ranges follow the data in view (0.5-99.5 percentiles, each sign of K
separately with K = 0 white) unless fixed with --chi/k/ham/weyl-range.
Reads plots/plt*, sahf_output/horizon_*.dat and punctures_output/punctures.dat.
Space freezes the view, q quits. Adapted from GRAttrezzo's BIBS vis2 plotters.
"""

import argparse
import re
from pathlib import Path

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import FuncNorm, LogNorm, Normalize
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError as missing:
    raise SystemExit(f"needs numpy and matplotlib ({missing.name} is missing); "
                     "try ~/.venvs/bibs-vis/bin/python") from None

BOX = re.compile(r"\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\)")

# fields built from plotfile variables: name -> (inputs, combine)
DERIVED = {
    "abs_Ham": (("Ham",), np.abs),
    "abs_Weyl4": (("Weyl4_Re", "Weyl4_Im"), np.hypot),
}

# layout in inches
SIDE = 4.2
BAR_GAP, BAR_WIDTH = 0.06, 0.13
COLUMN_GAP = 0.75
ROW_GAP = 0.45
LEFT, RIGHT, BOTTOM, TOP = 0.5, 0.75, 0.5, 0.45


# --------------------------------------------------------------------------
# plotfiles


class Plotfile:
    """AMReX HyperCLaw-V1.1 plotfile, 3D cell data."""

    def __init__(self, path):
        self.path = Path(path)
        lines = (self.path / "Header").read_text().splitlines()
        ncomp = int(lines[1])
        self.names = lines[2 : 2 + ncomp]
        i = 2 + ncomp + 1  # skip the dimension
        self.time = float(lines[i])
        self.finest_level = int(lines[i + 1])
        self.prob_lo = np.array(lines[i + 2].split(), dtype=float)
        self.prob_hi = np.array(lines[i + 3].split(), dtype=float)
        self.dx = [float(lines[i + 7 + lev].split()[0])
                   for lev in range(self.finest_level + 1)]

    def _fabs(self, level):
        """(index lo, index hi, file, offset) of every box on a level."""
        cell_h = (self.path / f"Level_{level}" / "Cell_H").read_text()
        boxes = [tuple(map(int, m.groups())) for m in BOX.finditer(cell_h)]
        files = re.findall(r"FabOnDisk: (\S+) (\d+)", cell_h)
        return [(np.array(b[:3]), np.array(b[3:]), name, int(offset))
                for b, (name, offset) in zip(boxes, files)]

    def z_slices(self, var, level, z):
        """Data at height z from every box that reaches it, linear between
        the two cell layers around z. Returns a list of (lo, hi, array[y, x])."""
        comp = self.names.index(var)
        kc = (z - self.prob_lo[2]) / self.dx[level] - 0.5  # cell-centre index
        k0 = int(np.floor(kc))
        w = kc - k0
        slices = []
        for lo, hi, name, offset in self._fabs(level):
            layers = [k for k in (k0, k0 + 1) if lo[2] <= k <= hi[2]]
            if not layers:
                continue
            with open(self.path / f"Level_{level}" / name, "rb") as fab:
                fab.seek(offset)
                header = fab.readline().decode()
                order = re.search(r"\(8, \(([\d ]+)\)\)\)", header)
                dtype = "<f8" if order.group(1).startswith("8") else ">f8"
                n = hi - lo + 1
                layer = n[0] * n[1]

                def read(k):
                    fab.seek(offset + len(header)
                             + 8 * (comp * layer * n[2] + (k - lo[2]) * layer))
                    return np.fromfile(fab, dtype, layer).reshape(n[1], n[0])

                data = ((1 - w) * read(k0) + w * read(k0 + 1)
                        if len(layers) == 2 else read(layers[0]))
            slices.append((lo, hi, data))
        return slices


def find_plotfiles(run_dir):
    """Complete plotfiles in order (AMReX writes pltXXXXX.temp, then renames)."""
    return sorted(p for p in (run_dir / "plots").glob("plt*")
                  if p.name[3:].isdigit() and (p / "Header").exists())


def inputs_of(field):
    return DERIVED[field][0] if field in DERIVED else (field,)


def level_image(plotfile, field, level, z):
    """NaN-padded array of a level's slice, its extent and its boxes."""
    per_input = [plotfile.z_slices(name, level, z) for name in inputs_of(field)]
    slices = per_input[0]
    if not slices:
        return None
    if field in DERIVED:
        combine = DERIVED[field][1]
        slices = [(lo, hi, combine(*(s[i][2] for s in per_input)))
                  for i, (lo, hi, _) in enumerate(slices)]
    lo = np.min([s[0][:2] for s in slices], axis=0)
    hi = np.max([s[1][:2] for s in slices], axis=0)
    image = np.full((hi[1] - lo[1] + 1, hi[0] - lo[0] + 1), np.nan)
    for box_lo, box_hi, data in slices:
        i0, j0 = box_lo[:2] - lo
        image[j0 : j0 + data.shape[0], i0 : i0 + data.shape[1]] = data
    dx = plotfile.dx[level]
    x0, y0 = plotfile.prob_lo[:2] + lo * dx
    x1, y1 = plotfile.prob_lo[:2] + (hi + 1) * dx
    return image, (x0, x1, y0, y1), [s[:2] for s in slices]


def draw_amr(ax, plotfile, level, image, boxes, style):
    """Outline a level (solid) and/or its boxes (dashed); style is "levels",
    "boxes" or "both"."""
    colour = matplotlib.colormaps["cool"](level / max(plotfile.finest_level, 1))
    dx = plotfile.dx[level]
    if style in ("levels", "both"):
        # the 0.5 contour of the padded "covered" mask runs along cell faces
        lo_index = np.min([b[0][:2] for b in boxes], axis=0)
        mask = np.pad(~np.isnan(image), 1).astype(float)
        lo = plotfile.prob_lo[:2] + (lo_index - 0.5) * dx
        xc = lo[0] + dx * np.arange(mask.shape[1])
        yc = lo[1] + dx * np.arange(mask.shape[0])
        ax.contour(xc, yc, mask, levels=[0.5], colors=[colour],
                   linewidths=0.9, alpha=0.9, zorder=51)
    if style in ("boxes", "both"):
        for box_lo, box_hi in boxes:
            x0, y0 = plotfile.prob_lo[:2] + box_lo[:2] * dx
            x1, y1 = plotfile.prob_lo[:2] + (box_hi[:2] + 1) * dx
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec=colour, lw=0.45, alpha=0.45,
                                   ls=(0, (4, 3)), zorder=50))


def slice_levels(plotfile, field, z):
    """(level, level_image) of every level that reaches z, coarse to fine."""
    return [(level, result) for level in range(plotfile.finest_level + 1)
            if (result := level_image(plotfile, field, level, z)) is not None]


def view_samples(results, view, n=300):
    """The composite field (finest level available) on an n x n grid over
    view = (x0, x1, y0, y1), i.e. what the panel shows."""
    xs = np.linspace(view[0], view[1], n)
    ys = np.linspace(view[2], view[3], n)
    out = np.full((n, n), np.nan)
    for _, (image, (x0, x1, y0, y1), _) in results:
        i = np.floor((xs - x0) / (x1 - x0) * image.shape[1]).astype(int)
        j = np.floor((ys - y0) / (y1 - y0) * image.shape[0]).astype(int)
        oi = (i >= 0) & (i < image.shape[1])
        oj = (j >= 0) & (j < image.shape[0])
        level = np.full((n, n), np.nan)
        level[np.ix_(oj, oi)] = image[np.ix_(j[oj], i[oi])]
        out = np.where(np.isnan(level), out, level)
    return out


def two_sided_symlog(vmin, vmax):
    """Symmetric-log norm with 0 at the middle of a diverging colour map and
    each sign stretched over its own half, linear within 1e-3 of the larger
    end. Returns the norm and colour-bar ticks (0 and decades)."""
    lin = 1e-3 * max(vmax, -vmin)
    vmin, vmax = min(vmin, -10 * lin), max(vmax, 10 * lin)

    def g(x):
        return np.sign(x) * np.log1p(np.abs(x) / lin)

    def g_inv(u):
        return np.sign(u) * lin * np.expm1(np.abs(u))

    gp, gn = g(vmax), -g(vmin)
    norm = FuncNorm((lambda x: np.where(x >= 0, g(x) / gp, g(x) / gn),
                     lambda y: np.where(y >= 0, g_inv(y * gp), g_inv(y * gn))),
                    vmin=vmin, vmax=vmax)
    ticks = [0.0]
    for sign, end in ((1, vmax), (-1, -vmin)):
        first = int(np.ceil(np.log10(10 * lin)))
        ticks += [sign * 10.0**k for k in range(first, int(np.floor(np.log10(end))) + 1)]
    return norm, sorted(ticks)


def colour_norm(kind, samples, fixed):
    """Norm (and ticks) of a panel: kind "lin", "log" or "div" (diverging
    about 0); the range is fixed, or the 0.5-99.5 percentiles of the samples
    (each sign separately for "div"). None if nothing to show."""
    if fixed is None:
        s = samples[np.isfinite(samples)]
        if kind == "log":
            s = s[s > 0]
        if not s.size:
            return None
        if kind == "div":
            pos, neg = s[s > 0], -s[s < 0]
            lo = -np.percentile(neg, 99.5) if neg.size else 0.0
            hi = np.percentile(pos, 99.5) if pos.size else 0.0
        else:
            lo, hi = np.percentile(s, [0.5, 99.5])
    else:
        lo, hi = fixed
    if kind == "div":
        return two_sided_symlog(lo, hi)
    if hi <= lo:
        lo, hi = (lo / 2, lo * 2) if kind == "log" else (lo - 1e-12, lo + 1e-12)
    return (LogNorm(lo, hi) if kind == "log" else Normalize(lo, hi)), None


# --------------------------------------------------------------------------
# text data (SmallDataIO files: '#' header lines, possibly a half-written
# last line)


def read_table(path, ncols):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if line.startswith("#") or len(parts) != ncols:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                pass
    return np.array(rows).reshape(-1, ncols)


HORIZON_COLUMNS = ["t", "x", "y", "z", "r_mean", "area", "M_A", "M_T_rot",
                   "chi_rot", "M_T_akv", "chi_akv", "Px", "Py", "Pz", "Sx_rot",
                   "Sy_rot", "Sz_rot", "Sx_akv", "Sy_akv", "Sz_akv", "rmsTheta",
                   "steps"]


def read_horizons(run_dir):
    """{id: {column: array}} from sahf_output/horizon_<id>.dat."""
    horizons = {}
    for path in sorted((run_dir / "sahf_output").glob("horizon_*.dat")):
        table = read_table(path, len(HORIZON_COLUMNS))
        horizons[int(path.stem.split("_")[1])] = dict(zip(HORIZON_COLUMNS,
                                                          table.T))
    return horizons


def punctures_at(run_dir, time):
    """Puncture positions [[x, y, z], ...] at a time, or None."""
    table = read_table(run_dir / "punctures_output" / "punctures.dat", 7)
    if not table.size:
        return None
    return np.array([[np.interp(time, table[:, 0], table[:, 1 + 3 * p + d])
                      for d in range(3)] for p in range(2)])


# --------------------------------------------------------------------------
# figure


def message(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.3", zorder=100,
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))


def build_figure(args):
    run_dir = args.run_dir

    def colour_map(name):
        cmap = matplotlib.colormaps[name].copy()
        cmap.set_bad(alpha=0.0)  # NaN padding lets coarser levels show through
        return cmap

    panels = [  # field, label, colour map, kind, fixed range, overlay colour
        ("chi", r"$\chi$", colour_map("magma"), "lin", args.chi_range, "white"),
        ("K", r"$K$", colour_map("RdBu_r"), "div", args.k_range, "black"),
        ("abs_Ham", r"$|\mathcal{H}|$", colour_map("turbo"), "log",
         args.ham_range, "white"),
        ("abs_Weyl4", r"$|\Psi_4|$", colour_map(args.weyl_cmap), "log",
         args.weyl_range, "white"),
    ]

    column = SIDE + BAR_GAP + BAR_WIDTH + COLUMN_GAP
    width = LEFT + 2 * column - COLUMN_GAP + RIGHT
    height = BOTTOM + 2 * (SIDE + ROW_GAP) + TOP
    fig = plt.figure(figsize=(width, height))

    def rect(row, col, dx=0.0, w=SIDE):
        x0 = LEFT + col * column + dx
        y0 = BOTTOM + (1 - row) * (SIDE + ROW_GAP)
        return [x0 / width, y0 / height, w / width, SIDE / height]

    axes, bars = [], []
    for k, (_, label, cmap, kind, _, _) in enumerate(panels):
        row, col = divmod(k, 2)
        ax = fig.add_axes(rect(row, col))
        ax.tick_params(labelsize=8, length=2, pad=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(label, fontsize=14, pad=5)
        if kind == "log":
            ax.set_facecolor(cmap(0.0))  # zeros are masked by the log
        axes.append(ax)
        bars.append(fig.add_axes(rect(row, col, SIDE + BAR_GAP, BAR_WIDTH)))

    def draw_bar(cax, norm, ticks, cmap):
        cax.clear()
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cax, extend="both")
        if ticks is not None:
            bar.set_ticks(ticks, labels=[
                "0" if t == 0 else
                rf"${'-' if t < 0 else ''}10^{{{round(np.log10(abs(t)))}}}$"
                for t in ticks])
        bar.ax.tick_params(labelsize=8)

    def draw(plotfile):
        punctures = punctures_at(run_dir, plotfile.time)
        if punctures is None:
            centre = 0.5 * (plotfile.prob_lo + plotfile.prob_hi)
            window = args.window or 20.0
        elif args.centre == "mid":
            centre = punctures.mean(axis=0)
            sep = np.linalg.norm(punctures[0, :2] - punctures[1, :2])
            window = args.window or max(4.0, 0.5 * sep + 4.0)
        else:
            centre = punctures[int(args.centre) - 1]
            window = args.window or 2.0

        horizons = read_horizons(run_dir)
        view = (centre[0] - window, centre[0] + window,
                centre[1] - window, centre[1] + window)
        for ax, cax, (field, _, cmap, kind, fixed, overlay) in zip(axes, bars,
                                                                  panels):
            for artist in [*ax.images, *ax.patches, *ax.texts, *ax.collections,
                           *ax.lines]:
                artist.remove()
            missing = [n for n in inputs_of(field) if n not in plotfile.names]
            results = [] if missing else slice_levels(plotfile, field, centre[2])
            scale = colour_norm(kind, view_samples(results, view), fixed)
            if missing or scale is None:
                message(ax, f"{' '.join(missing)} not in the plotfiles"
                        if missing else "no data in view")
            else:
                norm, ticks = scale
                for level, (image, extent, boxes) in results:
                    ax.imshow(image, origin="lower", extent=extent, norm=norm,
                              cmap=cmap, interpolation="nearest", zorder=level)
                    if args.amr and level > 0:
                        draw_amr(ax, plotfile, level, image, boxes, args.amr)
                draw_bar(cax, norm, ticks, cmap)
            if punctures is not None:
                ax.plot(punctures[:, 0], punctures[:, 1], "+", color=overlay,
                        ms=6, mew=1.0, zorder=60)
            for hid, h in horizons.items():
                near = np.where(np.abs(h["t"] - plotfile.time) < 0.1)[0]
                if near.size:
                    j = near[0]
                    ax.add_patch(Circle((h["x"][j], h["y"][j]), h["r_mean"][j],
                                        fill=False, ec=overlay, lw=1.2,
                                        zorder=61))
                    ax.text(h["x"][j], h["y"][j] + 1.15 * h["r_mean"][j],
                            f"H{hid}", color=overlay, fontsize=8, ha="center",
                            va="bottom", zorder=61)
            ax.set_xlim(centre[0] - window, centre[0] + window)
            ax.set_ylim(centre[1] - window, centre[1] + window)
            ax.set_aspect("equal", adjustable="datalim")
        fig.suptitle(rf"KerrBinary    $t = {plotfile.time:.2f}$    "
                     rf"{plotfile.path.name}    $z = {centre[2]:.2f}$",
                     fontsize=14, y=1.0 - 0.08 / height)

    return fig, draw


def watch(fig, run_dir, draw, poll):
    """Show the newest plotfile and jump to each new one. Space freezes, q
    quits."""
    state = {"paused": False}

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]

    fig.canvas.mpl_connect("key_press_event", on_key)
    shown = None
    if not find_plotfiles(run_dir):
        fig.suptitle(f"waiting for plotfiles in {run_dir.resolve()}")
    while plt.fignum_exists(fig.number):
        if not state["paused"]:
            plotfiles = find_plotfiles(run_dir)
            if plotfiles and plotfiles[-1] != shown:
                shown = plotfiles[-1]
                draw(Plotfile(shown))
        plt.pause(poll)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", default=".", type=Path,
                        help="run directory (default: current directory)")
    parser.add_argument("--centre", default="mid", choices=["mid", "1", "2"],
                        help="follow the punctures' midpoint or one puncture")
    parser.add_argument("--window", type=float, metavar="W",
                        help="half-width of the slice views (default: both "
                             "holes for mid, 2 for one hole)")
    parser.add_argument("--amr", default="levels",
                        choices=["levels", "boxes", "both", "none"],
                        help="overlay the refinement levels (default), boxes "
                             "or both")
    # colour ranges: by default the 0.5-99.5 percentiles of what is in view,
    # redone every frame (so colours are not comparable between frames)
    for name, what in (("chi", "chi"), ("k", "K (MIN < 0 < MAX)"),
                       ("ham", "|Ham|"), ("weyl", "|Psi4| = |Re + i Im|")):
        parser.add_argument(f"--{name}-range", type=float, nargs=2,
                            metavar=("MIN", "MAX"),
                            help=f"fixed colour range of {what} "
                                 "(default: from the data in view)")
    parser.add_argument("--weyl-cmap", default="turbo",
                        help="colour map of |Psi4| (default turbo)")
    parser.add_argument("--poll", type=float, default=1.0,
                        help="seconds between checks for new plotfiles "
                             "(default 1)")
    args = parser.parse_args()
    if args.amr == "none":
        args.amr = None

    fig, draw = build_figure(args)
    watch(fig, args.run_dir, draw, args.poll)


if __name__ == "__main__":
    main()
