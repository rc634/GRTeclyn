#!/bin/bash
# KerrBinary on Leonardo DCGP (2 x 56-core Sapphire Rapids, 512 GB per node), CPU build:
# 16 MPI ranks x 7 OpenMP threads (AMReX tiles each box over its rank's threads).
# Layout on scratch: $CINECA_SCRATCH/KERR-PROJ-DCGP/ holds the executable and the run
# directories (params.txt, initial_data.kid, this file as jobscript.slurm):
#   cd $CINECA_SCRATCH/KERR-PROJ-DCGP/leo_kerr && sbatch jobscript.slurm
#   sbatch --qos=dcgp_qos_dbg --time=00:30:00 jobscript.slurm   # a short first check
#   RESTART=chk00040 sbatch jobscript.slurm                      # continue from a checkpoint
# Rank r's stdout goes to pout/pout.r, errors to slurm-<jobid>.out.
# Params as on Booster, but amr.max_grid_size = 32 (enough boxes for 16 ranks) and no
# amrex.use_gpu_aware_mpi.
# Build (~/kerr-proj/GRTeclyn/Examples/KerrBinary): module restore kerrbin_dcgp;
# make -j 16 USE_CUDA=FALSE XTRA_CXXFLAGS=-march=sapphirerapids, then copy the .ex here.
#SBATCH --job-name=kerrbin
#SBATCH --account=INF26_teongrav_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=7
#SBATCH --time=24:00:00

module restore kerrbin_dcgp

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores OMP_PROC_BIND=close
srun --cpu-bind=cores ../KerrBinary3d.gnu.MPI.OMP.KerrBin.ex params.txt \
    ${RESTART:+amr.restart=checkpoints/$RESTART}
