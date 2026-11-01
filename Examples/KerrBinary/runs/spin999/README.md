# spin999: short chi >= 0.999 evolutions on the Leonardo DCGP debug queue (2026-11-01)

Tools for many short KerrBinary runs in `dcgp_qos_dbg` (30 min, 2 nodes, 2 jobs per user).
On Leonardo they live in `$CINECA_SCRATCH/KERR-PROJ-DCGP/spin999/`, with one directory per run.
The results are written up in kerrbin's CLAUDE.md ("Evolutions of chi 0.999 data on Leonardo").

- `make_par.sh name chi slice superposition p q m`: a kerrbin par in the leo_flyby_2 layout (A at rest with
  spin chi, B Schwarzschild M 0.5 at -24 moving at (0.27, 0.28)). Solve on the Mac: kerrbin with OpenBLAS on Leonardo
  ran 40-150x slower (8 solves sharing a node), and login nodes kill a solve after ~75 s of CPU.
- `single.template`, `binary.template`: params with @KEYS@ (MAXLEVEL, FLF, STOP, ETA, F0, GAUGE_EXTRA, NOTE).
  Coarsest dx 2 to +-128, z >= 0 half. `use_kerrbin = false` in single.template: hole A of the .kid alone.
- `mkrun.sh name kid template nodes KEY=value ...`: a run directory (params.txt, initial_data.kid, jobscript.slurm).
- `job.template`: runs every directory in `RUNS` at once (NT ranks each, `srun --exact --gres=none --mem=100G`:
  without --gres=none the job's default tmpfs gres goes to the first step and the others wait forever), each
  restarting from its newest checkpoint.
- `submit_next.sh`: submits the first line of `queue.txt` (`NAME NODES NT "RUN ..."`) whose runs are not in a queued
  job (run lists are stored as job comments), if fewer than 2 debug jobs are queued. Called every minute from the Mac.
- `spin_report.py RUN ...`: horizon table (r_mean, M_irr, M_T_akv, chi_akv), Ham/Mom norms outside the horizons
  from the plotfiles, and the Psi4 junk energy (high-passed above omega 0.2, window t - r < 40) per radius.
- `spin_plots.py -o out.png RUN:label ...`: r_mean, chi_akv - chi_akv(0), M_irr / M_irr(0) - 1 against t.

The executable was a scratch build (`spin999/build_flf`, outside the repo; `-I.` puts its headers first):
PunctureTagger accepts finest_level_factor >= 0.1 (repo: >= 1, too big for an LES Kerr horizon at r_+/4),
kid_file.cpp's geometry check at 1e-9 (since fixed in kerrbin), and KerrBinInitialData.hpp reads
`kerrbin.initial_lapse_power` (alpha_0 = chi^p, default 0.5 = the repo's sqrt(chi)).
