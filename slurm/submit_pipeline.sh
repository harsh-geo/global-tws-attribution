#!/bin/bash
# =========================================================================
# SLURM SUBMISSION SCRIPT FOR DIRAC SUPERCOMPUTER (IISER KOLKATA)
# PROJECT: Global Terrestrial Water Storage (TWS) Attribution Pipeline
# =========================================================================
#
# SBATCH DIRECTIVES
# -----------------
#SBATCH --job-name=tws_attribution_pipeline
#SBATCH --output=slurm/logs/tws_attr_%j.out
#SBATCH --error=slurm/logs/tws_attr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=standard

set -e # Exit immediately on script failure

echo "================================================================="
echo " Starting Global TWS Attribution SLURM Execution Pipeline"
echo " Job ID:            $SLURM_JOB_ID"
echo " Running on Node:   $SLURMD_NODENAME"
echo " Allocated Cores:   $SLURM_CPUS_PER_TASK"
echo " Start Time:        $(date)"
echo "================================================================="

# Create SLURM logging directory if not existing
mkdir -p slurm/logs

# Load MATLAB module on HPC cluster (DIRAC Supercomputer)
module purge
module load matlab/R2021b || module load matlab/R2022a || module load matlab

# Verify MATLAB availability
which matlab

# Set up project workspace root directory
PROJECT_DIR=$(pwd)
echo "Working directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Common Headless MATLAB Execution Flag Directive (GEMINI.md standard):
# -nodisplay -nosplash -nodesktop
MATLAB_RUN="matlab -nodisplay -nosplash -nodesktop -r"

# =========================================================================
# END-TO-END PIPELINE EXECUTION (In-Memory Processing)
# =========================================================================
echo ""
echo "-----------------------------------------------------------------"
echo "Executing full end-to-end TWS attribution pipeline..."
echo "-----------------------------------------------------------------"
$MATLAB_RUN "run('run_pipeline_local.m'); exit;"

echo ""
echo "================================================================="
echo " Pipeline Execution Completed Successfully!"
echo " End Time: $(date)"
echo " Results exported to outputs/tables/"
echo "================================================================="
