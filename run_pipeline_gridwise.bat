@echo off
echo =================================================================
echo   GLOBAL TWS ATTRIBUTION & TREND PIPELINE - GRID-WISE MLR BATCH
echo =================================================================
echo Starting MATLAB pipeline runner...
matlab -batch "run('run_pipeline_gridwise.m');"
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
