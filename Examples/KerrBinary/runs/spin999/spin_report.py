#!/usr/bin/env python3
"""spin999 study: one summary per run directory.

    python spin_report.py RUN [RUN ...] [--times 0 2 5 10 20] [--rmax 20] [--no-plots]

Per run:
  horizons   r_mean, M_irr (= M_A), M_T_akv, chi_akv, rms Theta at the requested times
             (nearest row), and how many finds failed
  plotfiles  L2 and max of Ham and |Mom| (Mom_i plotted) outside every horizon
             (r > 1.2 r_mean about its centre) and inside r < rmax about A, finest data only,
             split into a near shell (r < 3 about each hole) and the rest
  psi4       junk energy at each extraction radius: E = r^2 / (16 pi) sum_lm int |int Psi4_lm dt|^2 dt,
             the inner integral by fixed-frequency integration (omega0 = 0.2), m > 0 doubled
             (z -> -z symmetry), up to the last time; and max |r Psi4_22|
Writes RUN/report.json as well.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

BOX = re.compile(r"\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\)")
HCOLS = ["t", "x", "y", "z", "r_mean", "area", "M_A", "M_T_rot", "chi_rot", "M_T_akv", "chi_akv",
         "Px", "Py", "Pz", "Sx_rot", "Sy_rot", "Sz_rot", "Sx_akv", "Sy_akv", "Sz_akv", "rmsTheta", "steps"]


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


def horizons(run):
    out = {}
    for path in sorted((run / "sahf_output").glob("horizon_*.dat")):
        tab = read_table(path, len(HCOLS))
        # a restart rewrites rows after the checkpoint: keep the last row for each time
        if len(tab):
            _, idx = np.unique(tab[::-1, 0], return_index=True)
            tab = tab[::-1][idx]
        out[int(path.stem.split("_")[1])] = {c: tab[:, i] for i, c in enumerate(HCOLS)}
    return out


class Plotfile:
    def __init__(self, path):
        self.path = Path(path)
        lines = (self.path / "Header").read_text().splitlines()
        ncomp = int(lines[1])
        self.names = lines[2:2 + ncomp]
        i = 2 + ncomp + 1
        self.time = float(lines[i])
        self.finest = int(lines[i + 1])
        self.lo = np.array(lines[i + 2].split(), dtype=float)
        self.dx = [float(lines[i + 7 + lev].split()[0]) for lev in range(self.finest + 1)]

    def boxes(self, level):
        cell_h = (self.path / f"Level_{level}" / "Cell_H").read_text()
        boxes = [tuple(map(int, m.groups())) for m in BOX.finditer(cell_h)]
        files = re.findall(r"FabOnDisk: (\S+) (\d+)", cell_h)
        return [(np.array(b[:3]), np.array(b[3:]), name, int(off)) for b, (name, off) in zip(boxes, files)]

    def read(self, level, box, comps):
        lo, hi, name, off = box
        n = hi - lo + 1
        with open(self.path / f"Level_{level}" / name, "rb") as fab:
            fab.seek(off)
            header = fab.readline().decode()
            order = re.search(r"\(8, \(([\d ]+)\)\)\)", header)
            dtype = "<f8" if order.group(1).startswith("8") else ">f8"
            data = np.fromfile(fab, dtype, n.prod() * len(self.names))
        data = data.reshape(len(self.names), n[2], n[1], n[0])
        return {c: data[self.names.index(c)] for c in comps}


def constraint_norms(pf, centres, radii, rmax):
    """L2 and max of Ham and |Mom| on uncovered cells outside 1.2 r_H of every hole, r < rmax about centres[0]."""
    keys = [f"near{i + 1}" for i in range(len(centres))] + ["far"]
    acc = {k: [0.0, 0.0, 0.0] for k in keys}  # sum H^2 dV, sum M^2 dV, sum dV
    mx = {k: [0.0, 0.0] for k in keys}
    for lev in range(pf.finest + 1):
        dx = pf.dx[lev]
        finer = [(lo * 2, hi * 2 + 1) for lo, hi, _, _ in pf.boxes(lev + 1)] if lev < pf.finest else []
        for box in pf.boxes(lev):
            lo, hi = box[0], box[1]
            x0 = pf.lo + (lo + 0.5) * dx
            xmax = pf.lo + (hi + 0.5) * dx
            # skip boxes entirely beyond rmax
            near_pt = np.clip(centres[0], x0, xmax)
            if np.linalg.norm(near_pt - centres[0]) > rmax:
                continue
            d = pf.read(lev, box, ["Ham", "Mom1", "Mom2", "Mom3"])
            n = hi - lo + 1
            z, y, x = np.meshgrid(*[x0[k] + dx * np.arange(n[k]) for k in (2, 1, 0)], indexing="ij")
            keep = np.ones(z.shape, bool)
            for flo, fhi in finer:  # cells covered by the next level
                a, b = np.maximum(flo, lo), np.minimum(fhi, hi)
                if np.all(a <= b):
                    keep[a[2] - lo[2]:b[2] - lo[2] + 1, a[1] - lo[1]:b[1] - lo[1] + 1, a[0] - lo[0]:b[0] - lo[0] + 1] = False
            r0 = np.sqrt((x - centres[0][0]) ** 2 + (y - centres[0][1]) ** 2 + (z - centres[0][2]) ** 2)
            keep &= r0 < rmax
            nears = []
            for c, rh in zip(centres, radii):
                rc = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
                keep &= rc > 1.2 * rh
                nears.append(rc < 3.0)
            if not keep.any():
                continue
            H = d["Ham"]
            M2 = d["Mom1"] ** 2 + d["Mom2"] ** 2 + d["Mom3"] ** 2
            dv = dx ** 3
            anynear = np.zeros(z.shape, bool)
            sels = []
            for i, nr in enumerate(nears):
                sels.append((f"near{i + 1}", keep & nr & ~anynear))
                anynear |= nr
            sels.append(("far", keep & ~anynear))
            for key, sel in sels:
                if sel.any():
                    acc[key][0] += np.sum(H[sel] ** 2) * dv
                    acc[key][1] += np.sum(M2[sel]) * dv
                    acc[key][2] += sel.sum() * dv
                    mx[key][0] = max(mx[key][0], np.abs(H[sel]).max())
                    mx[key][1] = max(mx[key][1], np.sqrt(M2[sel].max()))
    out = {}
    for key in acc:
        s = acc[key]
        out[key] = dict(L2_H=np.sqrt(s[0] / s[2]) if s[2] else np.nan, L2_M=np.sqrt(s[1] / s[2]) if s[2] else np.nan,
                        max_H=mx[key][0], max_M=mx[key][1])
    return out


def ffi_energy(t, psi, r, wcut):
    """r^2/(16 pi) int |int psi dt|^2 dt of the part of psi above frequency wcut (drops the static
    near-zone field and the slow orbital part), on a Tukey-tapered window."""
    dt = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    p = np.interp(tu, t, psi.real) + 1j * np.interp(tu, t, psi.imag)
    n = len(tu)
    taper = np.ones(n)
    k = max(1, n // 10)
    taper[:k] = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    taper[-k:] = taper[:k][::-1]
    w = 2 * np.pi * np.fft.fftfreq(n, dt)
    spec = np.fft.fft((p - p.mean()) * taper)
    spec[np.abs(w) < wcut] = 0
    w[w == 0] = 1
    news = np.fft.ifft(spec / (-1j * w))
    return r * r / (16 * np.pi) * np.sum(np.abs(news) ** 2) * dt


def psi4(run, tjunk, wcut):
    """per radius: junk energy and peak amplitudes for t - r < tjunk"""
    files = sorted((run / "extraction_data").glob("weyl_extraction_mode_*.dat"))
    if not files:
        return {}
    res = {}
    for path in files:
        l, m = int(path.stem[-2]), int(path.stem[-1])
        radii = None
        for line in path.read_text().splitlines():
            if line.startswith("#") and "r =" in line:
                radii = np.array(line.split("=")[1].split(), dtype=float)[::2]
                break
        tab = read_table(path, 1 + 2 * len(radii))
        if len(tab) < 8:
            continue
        _, idx = np.unique(tab[::-1, 0], return_index=True)
        tab = tab[::-1][idx]
        for i, r in enumerate(radii):
            sel = tab[:, 0] < r + tjunk
            if sel.sum() < 8:
                continue
            ps = tab[sel, 1 + 2 * i] + 1j * tab[sel, 2 + 2 * i]
            e = ffi_energy(tab[sel, 0], ps, r, wcut) * (2 if m > 0 else 1)
            d = res.setdefault(float(r), {"E": 0.0, "tmax": float(tab[sel, 0][-1])})
            d["E"] += e
            if (l, m) == (2, 2):
                d["max_rpsi22"] = float(np.abs(r * ps).max())
            if (l, m) == (2, 0):
                d["max_rpsi20"] = float(np.abs(r * ps).max())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--times", nargs="+", type=float, default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40])
    ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--plotfiles", default="all", help="all, none, or a stride like 2")
    ap.add_argument("--tjunk", type=float, default=40.0, help="junk window: t - r < tjunk")
    ap.add_argument("--wcut", type=float, default=0.2, help="high-pass frequency of the junk energy")
    args = ap.parse_args()
    for run in args.runs:
        rep = {"run": run.name}
        print(f"\n===== {run.name}")
        hz = horizons(run)
        rep["horizons"] = {}
        for hid, h in hz.items():
            t = h["t"]
            print(f"  horizon {hid}: {len(t)} finds, t {t.min() if len(t) else 0:.2f}..{t.max() if len(t) else 0:.2f}")
            print("     t      r_mean     M_irr        M_T_akv      chi_akv      rmsTheta")
            rows = []
            for tt in args.times:
                if not len(t) or tt > t.max() + 0.2:
                    continue
                i = np.argmin(np.abs(t - tt))
                rows.append({c: float(h[c][i]) for c in ("t", "r_mean", "M_A", "M_T_akv", "chi_akv", "rmsTheta")})
                print(f"  {h['t'][i]:6.2f}  {h['r_mean'][i]:.5f}  {h['M_A'][i]:.7f}  {h['M_T_akv'][i]:.7f}  "
                      f"{h['chi_akv'][i]:.7f}  {h['rmsTheta'][i]:.2e}")
            if len(t):
                last = {c: float(h[c][-1]) for c in ("t", "r_mean", "M_A", "M_T_akv", "chi_akv", "rmsTheta")}
                rows.append(last)
                print(f"  last {last['t']:6.2f}  {last['r_mean']:.5f}  {last['M_A']:.7f}  {last['M_T_akv']:.7f}  "
                      f"{last['chi_akv']:.7f}  {last['rmsTheta']:.2e}")
            rep["horizons"][hid] = rows
        if args.plotfiles != "none":
            pfs = sorted(p for p in (run / "plots").glob("plt*") if p.name[3:].isdigit() and (p / "Header").exists())
            if args.plotfiles != "all":
                pfs = pfs[::int(args.plotfiles)]
            rep["constraints"] = []
            print("     t    region: L2 H     max H     L2 M     max M   (outside 1.2 r_H; near = r < 3 about a hole)")
            for p in pfs:
                pf = Plotfile(p)
                centres, radii = [], []
                for hid, h in sorted(hz.items()):
                    if len(h["t"]):
                        i = np.argmin(np.abs(h["t"] - pf.time))
                        centres.append(np.array([h["x"][i], h["y"][i], h["z"][i]]))
                        radii.append(h["r_mean"][i])
                if not centres:
                    continue
                c = constraint_norms(pf, centres, radii, args.rmax)
                rep["constraints"].append({"t": pf.time, **c})
                for key, v in c.items():
                    print(f"  {pf.time:6.2f}  {key:>6}: {v['L2_H']:.2e}  {v['max_H']:.2e}  {v['L2_M']:.2e}  {v['max_M']:.2e}")
        ps = psi4(run, args.tjunk, args.wcut)
        rep["psi4"] = ps
        for r, d in sorted(ps.items()):
            print(f"  psi4 r = {r:5.1f}: E_junk(t < {d['tmax']:.1f}, omega > {args.wcut}) = {d['E']:.3e}   max|r psi4_22| = "
                  f"{d.get('max_rpsi22', np.nan):.3e}   max|r psi4_20| = {d.get('max_rpsi20', np.nan):.3e}")
        (run / "report.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
