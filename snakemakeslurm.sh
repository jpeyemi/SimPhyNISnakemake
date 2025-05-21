#!/bin/bash

#
# check for clean
#
# https://snakemake.readthedocs.io/en/stable/project_info/faq.html#how-do-i-remove-all-files-created-by-snakemake-i-e-like-make-clean
# if [ "$1" == "clean" ]; then
#     echo 'rm $(snakemake --summary | tail -n+2 | cut -f1)'
#     snakemake --summary | tail -n+2 | cut -f1
#     rm -f $(snakemake --summary | tail -n+2 | cut -f1)
#     exit 0
# fi

#
# launch snakemake to run jobs via SLURM
# ntasks
SM_PARAMS="job-name ntasks partition time mail-user mail-type error output"

SM_ARGS=" --no-requeue --parsable --cpus-per-task {cluster.cpus-per-task} --mem {cluster.mem}"

for P in ${SM_PARAMS}; do SM_ARGS="$SM_ARGS --$P {cluster.$P}"; done
echo "SM_ARGS: ${SM_ARGS}"

# our SLURM error/output paths expect a logs/ subdir in PWD
mkdir -p logs

conda config --set ssl_verify no

### run snakemake
# -j defines total number of jobs executed in parallel on cluster
# -n dryrun
# -p print command lines
# --use-conda allows to activate conda env necessary for rule
# --conda-prefix envs: builds environment in envs folder where it will be 
snakemake -p \
    $* \
     --latency-wait 120 \
    -j 490 \
    --cluster-config $(dirname $0)/cluster.slurm.json \
    --cluster "sbatch $SM_ARGS" \
    --cluster-status /home/iobal/mit_lieberman/scripts/slurm_status.py \
    --rerun-incomplete \
    --restart-times 2 \
    --keep-going \
    --use-conda \
    --conda-frontend conda \
    --conda-prefix /home/iobal/mit_lieberman/tools/conda_snakemake \
    -s Snakefile.py 
    # --dag \
    # | dot -Tsvg > dag.svg

    # --groups make_blastdb_refgenome=smalljobs blast_refgenome=smalljobs --group-components smalljobs=10 \