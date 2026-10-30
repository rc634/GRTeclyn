# KerrBinary: initial data from kerrbin

Evolves two black holes (boosted Kerr or Schwarzschild) from initial data
solved by [kerrbin](../../../kerrbin): the CTT constraint equations on a
bispherical spectral grid, with each hole's exact slice inside its excision
sphere. kerrbin writes the solution to a `.kid` file; this example reads it.

## Build

kerrbin is found at `KERRBIN_HOME`, by default `../kerrbin` next to GRTeclyn.
Only its evaluation code is compiled (no solver, no LAPACK), through
`Tools/GNUMake/packages/Make.kerrbin` (`USE_KERRBIN = TRUE` in the GNUmakefile).

The SAHF horizon finders (`USE_SAHF = TRUE`, `Tools/GNUMake/packages/Make.sahf`)
need SAHF's `akv` branch at `SAHF_HOME`, by default `../SAHF`, and **LAPACK**
for the approximate Killing vector spin: `SAHF_LAPACK`, by default `-llapack`
(with an OpenBLAS module, e.g. `SAHF_LAPACK="-L$OPENBLAS_LIB -lopenblas"`).
Set `USE_SAHF = FALSE` to build without them.

```bash
make -j 8                                                          # CPU
make -j 16 COMP=gnu USE_CUDA=TRUE AMREX_CUDA_ARCH=8.0 USE_OMP=TRUE # A100s
```

## Run

Each run lives in its own directory under `runs/`, with the `.kid` file next
to the parameter file:

```bash
cd runs/flyby_sahf && ./run.sh                    # Mac: mpirun -n 4
cd runs/leo_kerr && sbatch ../leonardo.slurm      # Leonardo: 1 node, 4 A100
```

`kerrbin.kid_file` is the path of the `.kid` file, relative to the run
directory. The `.kid` frame (the frame of kerrbin's parameter file) has its
origin at `geometry.center`. The puncture tracker starts from the hole
positions in the file, and puncture tagging uses the bare masses.
`kerrbin.only_A = true` evolves hole A of the file alone, with its exact slice
everywhere (`runs/leo_kerr`, the control run of `runs/leo_flyby`).

`sahf.*` parameters (see `runs/leo_flyby/params.txt`) run `sahf.n` horizon
finders on every time step of `sahf.level` and print each horizon (mass, spin,
momentum) to stdout and to `sahf_output/horizon_<id>.dat`.

Rank r's stdout goes to `pout/pout.r` (`grteclyn.pout = false`: the terminal).
On a restart (`amr.restart`), the logs are appended to, the puncture, Weyl and
horizon files lose their rows after the restart time (they are redone), and
each horizon finder starts from its last row at or before it.

## Initial data fill

kerrbin is CPU code, so the data are evaluated on the host
(`Source/InitialData/KerrBin/KerrBinInitialData.hpp`): each MPI rank fills its
own boxes, OpenMP threads share the tiles of each box (also in GPU builds), and
GPU builds copy the result to the device once. Gamma^i then comes from finite
differences of h_ij (`GammaCalculator`). The cost is about 75 us per point
outside the excision spheres at p, q = 26, m = 10 on one core (more at higher
kerrbin resolution), so use `USE_OMP=TRUE` and `OMP_NUM_THREADS` to fill the
cores that the GPU ranks leave idle.
