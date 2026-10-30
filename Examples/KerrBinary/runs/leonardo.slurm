#!/bin/bash
# KerrBinary on Leonardo Booster (4 A100 per node): one MPI rank per GPU, the node's
# other cores as OpenMP threads for the kerrbin initial data fill (host code).
# Submit from a run directory:
#   cd runs/leo_kerr && sbatch ../leonardo.slurm
#   sbatch --qos=boost_qos_dbg --time=00:30:00 ../leonardo.slurm   # a short first check
# Rank r's stdout goes to pout/pout.r, errors to slurm-<jobid>.out.
#SBATCH --job-name=kerrbin
#SBATCH --account=FILL_IN              # saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

module load FILL_IN                    # the modules the executable was built with

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpu-bind=cores ../../KerrBinary3d.gnu.MPI.OMP.CUDA.KerrBin.ex params.txt
