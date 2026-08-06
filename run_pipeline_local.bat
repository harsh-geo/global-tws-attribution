@echo off
echo =================================================================
echo   GLOBAL TWS ATTRIBUTION & TREND PIPELINE - LOCAL BATCH EXECUTION
echo =================================================================
echo Starting MATLAB pipeline runner...
matlab -batch "run('run_pipeline_local.m');"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo =================================================================
    echo   Pipeline completed successfully!
    echo =================================================================
) else (
    echo.
    echo [ERROR] Pipeline execution failed with exit code %ERRORLEVEL%.
)
pause
