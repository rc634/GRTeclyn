#!/bin/bash
# mkrun.sh name kid template nodes key=value ...   (keys: NOTE MAXLEVEL FLF STOP ETA GAUGE_EXTRA F0 ...)
name=$1; kid=$2; tmpl=$3; nodes=$4; shift 4
mkdir -p $name && cp $kid $name/initial_data.kid
cp $tmpl $name/params.txt
for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}; v=${v//\//\\/}; sed -i "s/@$k@/$v/g" $name/params.txt; done
sed -i "s/@GAUGE_EXTRA@//; s/@NOTE@//" $name/params.txt
sed "s/@NAME@/$name/; s/@NODES@/$nodes/" job.template > $name/jobscript.slurm
grep -n "@" $name/params.txt && echo "UNFILLED in $name" || echo "made $name"
