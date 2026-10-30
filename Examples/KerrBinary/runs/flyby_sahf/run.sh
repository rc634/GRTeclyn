#!/bin/bash
# Run this directory's params.txt on NP ranks (default 4). Rank r's output goes
# to pout/pout.r (appended, so a restart continues the log): tail -f pout/pout.0
cd "$(dirname "$0")" || exit 1
mpirun -n "${NP:-4}" ../../KerrBinary3d.llvm.MPI.KerrBin.ex params.txt
