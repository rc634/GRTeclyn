# GRTeclyn — working notes for Claude

## Target platforms
- **Local (Mac, Intel i7, AMD Radeon Pro 580):** CPU only — no usable GPU backend. Build `COMP=llvm USE_MPI=TRUE` (Apple clang 15 + Open MPI 5.0.1). Used for low-res tests only.
- **Production (Leonardo Booster, CINECA):** NVIDIA A100 64GB, 4 per node. Build `COMP=gnu USE_MPI=TRUE USE_CUDA=TRUE AMREX_CUDA_ARCH=8.0`, one MPI rank per GPU, CUDA-aware OpenMPI.
- AMReX lives at `$AMREX_HOME=/Users/admin/Documents/robin/amrex` (not side by side with this repo). Keep GRTeclyn and AMReX on the same commits on both machines.

## GPU safety — enforce on every change
All code must build and run correctly with `USE_CUDA=TRUE`, even though it is only tested on CPU locally. On the Mac, `AMREX_GPU_DEVICE` expands to nothing, so the compiler will NOT catch these. When writing or reviewing code, check every item and flag violations to the user before they become a problem on Leonardo.

### Inside `amrex::ParallelFor` lambdas
- Lambda is `[=] AMREX_GPU_DEVICE (...)`. Never `[&]`.
- No implicit `this`: never use member variables (`m_*`) or call member functions inside the lambda. Copy to a local `const` first.
- Every function/method called inside is marked `AMREX_GPU_HOST_DEVICE` (or `AMREX_GPU_DEVICE`), including in compute classes and helpers.
- Captured objects are trivially copyable: no `std::vector`, `std::string`, `std::map`, `std::function`, `std::unique_ptr` members. Use `amrex::GpuArray`, plain scalars, fixed-size arrays. Add `static_assert(std::is_trivially_copyable_v<T>)` for new compute classes (this one DOES fail on the Mac).
- Not allowed: `std::cout`, exceptions, `new`/`malloc`, virtual calls, `GRParmParse`/`pp.get`, non-`constexpr` globals/statics (e.g. `StateVariables::names`), recursion.
- No shared accumulators (`sum += ...`). Use `amrex::ReduceOps`/`ReduceData` or `amrex::Gpu::Atomic::Add`.
- No in-place stencil updates (read neighbours of a field written in the same kernel).

### Enclosing functions
- Functions containing a device lambda must be `public` (nvcc rejects lambdas in private/protected member functions).
- Read parameters, build compute classes, and get `arrays()` views outside the lambda.

### Host code touching grid data
- Never dereference `MultiFab`/`Array4` data in host code (diagnostics, small-data IO, debug prints, AH finder, surface integrals). On GPU it is device memory. Use reductions, `ParticleInterpolator`, or an explicit host copy (`The_Cpu_Arena` + `htod/dtoh_memcpy`, as in `BinaryBHLevel::initData` for TwoPunctures).
- Call `amrex::Gpu::streamSynchronize()` before host code uses results of a kernel.
- CPU-only libraries (TwoPunctures, GSL, etc.) must never be called inside kernels and should not run every timestep.

### Performance (works but slow on GPU)
- Don't merge large RHS computations into one giant kernel; follow the CCZ4RHS split-pass pattern.
- Avoid per-timestep host<->device copies and many tiny kernels.
- GPU params need much larger `amr.max_grid_size`/`amr.blocking_factor` than CPU tests, plus `amrex.the_arena_init_size` and `amrex.use_gpu_aware_mpi = 1`.

### Testing habits
- Run locally with `mpirun -n 4` (not only serial) to exercise box boundaries/ghost exchange.
- Compare Mac vs GPU results with tolerances, not bitwise.
- Remind the user to do a `USE_CUDA=TRUE` build on Leonardo after adding any new compute class or diagnostic, before long runs.

## kerrbin initial data (branch `kerrbin`)
- `Examples/KerrBinary`: two holes from a kerrbin `.kid` file (`kerrbin.kid_file`, relative to the run directory; runs in `runs/<name>/` with the `.kid` next to `params.txt`). The `.kid` frame origin is `geometry.center`.
- `Tools/GNUMake/packages/Make.kerrbin` (`USE_KERRBIN = TRUE`) compiles only kerrbin's evaluation sources from `KERRBIN_HOME` (default `../kerrbin`); no LAPACK. The source list and folders come from kerrbin's `evaluation.mk`, so new kerrbin modules (slices, interior fills) need no change here; it needs a kerrbin with `evaluation.mk` (2026-11-01 or later).
- Options that live in the `.kid` (chosen in kerrbin's par file, evaluated by kerrbin's code, nothing to set here): each hole's slice (`slice = les`: Liu-Etienne-Shapiro radius, horizon at r_+/4, for chi >~ 0.99) and the interior fill (`[interior] fill = kelvin`: no constraint shell at the excision spheres). Old `.kid` files read as before (quasi-isotropic, exact fill).
- `Source/InitialData/KerrBin/KerrBinInitialData.hpp`: host fill (kerrbin is CPU code: never call it in a kernel), MPI over boxes, OpenMP over explicit 8x8 tiles even in GPU builds, then one htod copy; Gamma^i by `GammaCalculator`. ~75-115 us per point per core, so build with `USE_OMP=TRUE` on Leonardo. The OpenMP path has not run yet (Apple clang has no `-fopenmp`).
- `use_kerrbin = false` (initial data type; default true = the binary solution): exact Kerr, hole A of the `.kid` alone, its exact slice everywhere (kerrbin's analytic `BoostedHole`, built in the user frame; matches the evaluator inside A's sphere to 2e-10). B's puncture is put on A's: `BHAmr<2>` is compile time.
- SAHF (`USE_SAHF`, SAHF `akv` branch, LAPACK via `SAHF_LAPACK`): runs on all ranks (every rank flows the same surfaces), writes `sahf_output/horizon_<id>.dat` on rank 0. On restart each finder starts from its last row at or before the restart time; SmallDataIO drops the later rows. The flow converges to a truncation floor, not zero (dx 1/64: rms Θ 2e-4 at t = 0, up to 2.06e-3 at t ~ 9 in the gauge transient), so it runs until it stalls there (rms down < 1% in 3 steps, counted only below `sahf.tol`) or for `max_iter` steps, and `sahf.tol` (1e-2) is only the acceptance ceiling. `sahf.adaptive_step` (default true) halves the flow step when an update reverses the previous one: at chi 0.999 (LES) a fixed `ffA = 0.5` oscillates and never converges (rms Θ stuck at 2.8); adaptive finds it in 137 steps cold, 27-42 warm, and at chi 0.99 the steps and results are unchanged. `false` restores the fixed step.
- `mainSetup` sends each rank's stdout to `<output_path>/pout/pout.<rank>` (appended; `grteclyn.pout = false` to keep the terminal). SAHF prints with printf, so this is a freopen of stdout, not an AMReX stream.
- Leonardo runs: `Examples/KerrBinary/runs/leo_kerr` (A alone) and `leo_flyby`, same grid (dx 1/64 finest, boundary 512, extraction 40-160 at dx 1), `runs/leonardo.slurm` (account INF26_teongrav, `module restore kerrbin`).
- Leonardo setup (rcroft00): `~/amrex` (same commit as the Mac) with `Tools/GNUMake/Make.local-pre` making GPU builds the default (gnu, MPI, CUDA arch 80, OpenMP, `SAHF_LAPACK = -lopenblas`); `AMREX_HOME` in `~/.bashrc`; `~/kerr-proj/{GRTeclyn,SAHF,kerrbin}`; module collection `kerrbin` = gcc/12.2.0 cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 openblas/0.3.26--gcc--12.2.0. Runs in `$CINECA_SCRATCH/KERR_PROJ/` (the executable, run directories with `jobscript.slurm`). Login: `login01-ext.leonardo.cineca.it` (the known host key).
- Leonardo DCGP (CPU, 2 x 56-core Sapphire Rapids per node): `runs/leonardo_dcgp.slurm` (account INF26_teongrav_0, 16 ranks x 7 threads per node, `module restore kerrbin_dcgp` = gcc, the same OpenMPI and OpenBLAS, no CUDA); build `make -j 16 USE_CUDA=FALSE XTRA_CXXFLAGS=-march=sapphirerapids`; params as on Booster but `amr.max_grid_size = 32` and no `amrex.use_gpu_aware_mpi`; runs in `$CINECA_SCRATCH/KERR-PROJ-DCGP/`. leo_kerr: 15.8 M/h on 1 node, 27.5 M/h on 2 (1.74x for 15% more core hours).
