#!/bin/bash
# Delete this run's output (named items only, after confirmation);
# keeps params.txt, initial_data.kid, kerrbin.par and this script.
cd "$(dirname "$0")" || exit 1

found=()
for f in plots plots.visit checkpoints extraction_data punctures_output \
         sahf_output parameters_and_version.txt pout pout.0 log.txt Backtrace.* \
         slurm-*.out; do
    [ -e "$f" ] && found+=("$f")
done
[ ${#found[@]} -eq 0 ] && { echo "nothing to wipe in $PWD"; exit 0; }

echo "in $PWD:"
printf '  %s\n' "${found[@]}"
read -r -p "delete these? [y/N] " a
if [ "$a" = y ]; then
    rm -rf -- "${found[@]}" && echo "wiped"
else
    echo "nothing deleted"
fi
