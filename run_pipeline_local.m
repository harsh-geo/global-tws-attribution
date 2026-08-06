%% run_pipeline_local.m - Master Local Pipeline Runner
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   Sequentially executes all preprocessing, gap-filling, attribution modeling,
%   validation, and visualization steps locally.
% =========================================================================

clear; clc;
t_start = tic;
fprintf('=================================================================\n');
fprintf('  GLOBAL TWS ATTRIBUTION & TREND PIPELINE - LOCAL EXECUTION     \n');
fprintf('=================================================================\n\n');

% Locate Project Root Directory
script_dir   = fileparts(mfilename('fullpath'));
project_root = script_dir;
if isempty(project_root)
    project_root = pwd;
end

% Add project subdirectories to MATLAB path
addpath(genpath(fullfile(project_root, 'src')));
addpath(genpath(fullfile(project_root, 'visualization')));

% --- CONFIGURATION ---
% Choose the attribution modeling approach for STEP 4:
% 'absolute' : Use raw absolute values for drivers (default)
% 'anomaly'  : Use anomaly values (mean removed) for all drivers
attribution_model = 'absolute'; 


%% STEP 1: Unit Standardization & Data Sanitization
fprintf('--> Running STEP 1: Hydroclimate Flux Standardization...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'preprocessing', 'step01_unit_conversion.m'));
fprintf('✓ STEP 1 Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 2: Latitude Cosine-Weighted Basin Aggregation
fprintf('--> Running STEP 2: Basin Aggregation across 103 Basins...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'preprocessing', 'step02_aggregate_basins.m'));
fprintf('✓ STEP 2 Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 3: Random Forest GRACE Gap-Filling
fprintf('--> Running STEP 3: Random Forest GRACE Reconstruction...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'gap_filling', 'step03_reconstruct_grace.m'));
fprintf('✓ STEP 3 Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 4: Twin RF Machine Learning Attribution
fprintf('--> Running STEP 4: Twin RF Attribution Modeling...\n');
t_step = tic;
if strcmpi(attribution_model, 'anomaly')
    fprintf('    Using ANOMALY values for predictors.\n'); %#ok<UNRCH>
    run(fullfile(project_root, 'src', 'modeling', 'step04b_run_attribution_anomalies.m')); %#ok<UNRCH>
else
    fprintf('    Using ABSOLUTE values for predictors.\n');
    run(fullfile(project_root, 'src', 'modeling', 'step04_run_attribution.m'));
end
fprintf('✓ STEP 4 Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 5: Validation & Trend Analysis
fprintf('--> Running STEP 5: Block CV Validation & Mann-Kendall Trends...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'validation', 'step05_validate_and_trends.m'));
fprintf('✓ STEP 5 Completed in %.2f seconds.\n\n', toc(t_step));

%% Summary
fprintf('=================================================================\n');
fprintf('  FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN %.2f SECS    \n', toc(t_start));
fprintf('  Outputs saved to: %s\n', fullfile(project_root, 'outputs'));
fprintf('=================================================================\n');
