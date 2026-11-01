#!/bin/bash
# on Leonardo, in spin999/: submit the first eligible line of queue.txt if fewer than 2 debug jobs are queued.
# queue.txt lines: JOBNAME NODES NT "RUN1 RUN2 ..."   (a line is eligible when none of its runs is in a queued job)
cd $CINECA_SCRATCH/KERR-PROJ-DCGP/spin999
[ -s queue.txt ] || exit 0
n=$(squeue -u rcroft00 -h -q dcgp_qos_dbg | wc -l)
[ "$n" -ge 2 ] && exit 0
busy=$(squeue -u rcroft00 -h -o "%k" | tr ' ' '\n')   # job comments hold the run lists
i=0
while IFS= read -r line; do
  i=$((i+1))
  [ -z "$line" ] && continue
  name=$(echo "$line" | awk '{print $1}'); nodes=$(echo "$line" | awk '{print $2}'); nt=$(echo "$line" | awk '{print $3}')
  runs=$(echo "$line" | cut -d'"' -f2)
  clash=0; for r in $runs; do echo "$busy" | grep -qx "$r" && clash=1; done
  [ $clash -eq 1 ] && continue
  sed "s/@NAME@/$name/; s/@NODES@/$nodes/" job.template > job_$name.slurm
  id=$(RUNS="$runs" NT=$nt sbatch --parsable --export=ALL --comment="$runs" job_$name.slurm)
  echo "$(date +%T) submitted $id $name: $runs" | tee -a queue.log
  sed -i "${i}d" queue.txt
  exit 0
done < queue.txt
