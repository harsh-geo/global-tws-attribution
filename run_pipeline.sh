#!/bin/bash
# Global TWS Attribution Pipeline Runner
# This script runs the entire preprocessing and modeling pipeline sequentially.
# 
# To run in the background and monitor:
#   nohup bash run_pipeline.sh > pipeline_output.log 2>&1 &
#   tail -f pipeline_output.log

# Find MATLAB executable (handles the nohup PATH issue on Mac)
MATLAB_EXE="matlab"
if ! command -v "$MATLAB_EXE" &> /dev/null; then
    # Search for MATLAB in the standard Mac Applications folder
    for app in /Applications/MATLAB_R*.app; do
        if [ -x "$app/bin/matlab" ]; then
            MATLAB_EXE="$app/bin/matlab"
            break
        fi
    done
fi

if ! command -v "$MATLAB_EXE" &> /dev/null; then
    echo "[ERROR] MATLAB executable not found."
    exit 1
fi

echo "================================================="
echo " Starting Global TWS Attribution Pipeline"
echo " Using MATLAB: $MATLAB_EXE"
echo " Start Time: $(date)"
echo "================================================="

# Array of steps to run
STEPS=(
    "src/preprocessing/step01_unit_conversion.m"
    "src/preprocessing/step02_aggregate_basins.m"
    "src/preprocessing/step02c_append_new_drivers.m"
    "src/modeling/step04_run_attribution.m"
)

for step in "${STEPS[@]}"; do
    echo ""
    echo ">>> Running $step at $(date)"
    
    # Execute the script in batch mode
    "$MATLAB_EXE" -nodesktop -nosplash -nodisplay -batch "run('$step');"
    
    # Check if the step succeeded
    if [ $? -ne 0 ]; then
        echo "[ERROR] $step failed. Halting pipeline."
        exit 1
    fi
    echo ">>> $step completed successfully."
done

echo ""
echo "================================================="
echo " Pipeline Finished Successfully at $(date)"
echo "================================================="
