#!/bin/bash
# make_par.sh name chiA sliceA superposition p q m  -> name.par (leo_flyby_2 layout, A chi along z)
name=$1; chi=$2; slice=$3; sup=$4; p=$5; q=$6; m=$7
cat > $name.par <<EOP
# spin999 study: leo_flyby_2 layout, A chi $chi ($slice), superposition $sup, p q m = $p $q $m
[hole A]
mass     = 1.0
position = 0, 0, 0
velocity  = 0, 0, 0
chi       = $chi
spin_axis = 0, 0, 1
sphere   = 0.8
slice    = $slice
[hole B]
mass     = 0.5
position = -24, 0, 0
velocity  = 0.27, 0.28, 0
chi       = 0
spin_axis = 0, 0, 1
sphere   = 0.8
[free data]
weight_power   = 4
metric_falloff = 0
superposition  = $sup
[spectral basis]
p_max = $p
q_max = $q
m_max = $m
[solver]
tolerance  = 1e-10
max_passes = 20
[horizons]
l_max = 12
[output]
dir = $name
kid = $name/initial_data.kid
phi = 0
EOP
