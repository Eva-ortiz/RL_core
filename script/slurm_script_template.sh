#!/bin/bash

## tng nodes info
## node-[04,07,09] : CPU, 64Cores, 512GB RAM
## node-[18-25] : CPU, 56 Cores, 128GB RAM, 2TB /scratch por nodo

#SBATCH --nodelist=##node to work with ex.: node-02 (CHANGE)
#SBATCH --ntasks=1
#SBATCH --get-user-env            ## Exports all local SHELL vars
#SBATCH --time=72:00:00 ##(CHANGE)
#SBATCH --cpus-per-task=##number_of_cpus ex.: 32 (CHANGE)
#SBATCH --mem-per-cpu=##RAM per cpu ex.:4GB (CHANGE)
#SBATCH --partition=##partition name (CHANGE)
##SBATCH --qos=long ##uncomment for runs longer than 10 days
#SBATCH --job-name=##JOB_NAME (CHANGE)
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# [DO NOT CHANGE - MANDATORY]
# Use '/scratch' FS - file system
# -------------------------------
source scratchfs
# -------------------------------
# SYNC DATA: uwd SlurmJobName.out
# -------------------------------

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
## export VIRTUAL_ENV=venv      ## venv instead of .venv

## -------- BEFORE SEND IT TO THE NODE  --------
## Go to the folder from which we want to execute our code (check current with pwd)

## Check desired environment is activated (terminal), else;
## source .venv/bin/activate

## If .venv not copied to computing nodes, perform following steps before code execution,
## - Change name of environment variable by uncomment ```export VIRTUAL_ENV=venv``` (see above)
## - Sync the active environment with ```uv sync --active``` (terminal)
## - Activate the environment with ```source venv/bin/activate``` (terminal)

## ------------------ EXECUTE  ------------------
## Execute our code with sbatch: sbatch ./script/slurm_script.sh
python ./RLcore/main.py ## normally run the code (CHANGE)