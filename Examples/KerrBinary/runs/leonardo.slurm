#!/bin/bash
# KerrBinary on Leonardo Booster (4 A100 per node): one MPI rank per GPU, the node's
# other cores as OpenMP threads for the kerrbin initial data fill (host code).
# Layout on scratch: $CINECA_SCRATCH/KERR_PROJ/ holds the executable and the run
# directories (params.txt, initial_data.kid, this file as jobscript.slurm):
#   cd $CINECA_SCRATCH/KERR_PROJ/leo_kerr && sbatch jobscript.slurm
#   sbatch --qos=boost_qos_dbg --time=00:30:00 jobscript.slurm   # a short first check
#   RESTART=chk00040 sbatch jobscript.slurm                       # continue from a checkpoint
# Rank r's stdout goes to pout/pout.r, errors to slurm-<jobid>.out.
# Build (~/kerr-proj/GRTeclyn/Examples/KerrBinary): module restore kerrbin; make -j 16
# (GPU options in $AMREX_HOME/Tools/GNUMake/Make.local-pre), then copy the .ex here.
#SBATCH --job-name=kerrbin
#SBATCH --account=INF26_teongrav
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

module restore kerrbin

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpu-bind=cores ../KerrBinary3d.gnu.MPI.OMP.CUDA.KerrBin.ex params.txt \
    ${RESTART:+amr.restart=checkpoints/$RESTART}
