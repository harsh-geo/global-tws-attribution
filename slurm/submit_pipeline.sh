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
# STEP 1: Preprocessing & Unit Conversion (cm/month standardization)
# =========================================================================
echo ""
echo "-----------------------------------------------------------------"
echo "[STEP 1/5] Running step01_unit_conversion.m..."
echo "-----------------------------------------------------------------"
$MATLAB_RUN "addpath(genpath('src')); run('src/preprocessing/step01_unit_conversion.m'); exit;"

# =========================================================================
# STEP 2: Latitude Cosine-Weighted Basin Spatial Aggregation (103 Basins)
# =========================================================================
echo ""
echo "-----------------------------------------------------------------"
echo "[STEP 2/5] Running step02_aggregate_basins.m..."
echo "-----------------------------------------------------------------"
$MATLAB_RUN "addpath(genpath('src')); run('src/preprocessing/step02_aggregate_basins.m'); exit;"

# =========================================================================
# STEP 3: Parallel Random Forest GRACE Gap Reconstruction (2002-2021)
# =========================================================================
echo ""
echo "-----------------------------------------------------------------"
echo "[STEP 3/5] Running step03_reconstruct_grace.m..."
echo "-----------------------------------------------------------------"
$MATLAB_RUN "addpath(genpath('src')); run('src/gap_filling/step03_reconstruct_grace.m'); exit;"

# =========================================================================
# STEP 4: Twin RF Machine Learning Attribution (TWSC = M_nat vs M_anthro)
# =========================================================================
echo ""
echo "-----------------------------------------------------------------"
echo "[STEP 4/5] Running step04_run_attribution.m..."
echo "-----------------------------------------------------------------"
$MATLAB_RUN "addpath(genpath('src')); run('src/modeling/step04_run_attribution.m'); exit;"

# =========================================================================
# STEP 5: 3-Year Block CV Validation & Modified Mann-Kendall Trend Testing
# =========================================================================
echo ""
echo "-----------------------------------------------------------------"
echo "[STEP 5/5] Running step05_validate_and_trends.m..."
echo "-----------------------------------------------------------------"
$MATLAB_RUN "addpath(genpath('src')); run('src/validation/step05_validate_and_trends.m'); exit;"

echo ""
echo "================================================================="
echo " Pipeline Execution Completed Successfully!"
echo " End Time: $(date)"
echo " Results exported to outputs/tables/"
echo "================================================================="
