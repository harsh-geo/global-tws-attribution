%% run_pipeline_gridwise.m - Master Grid-wise Pipeline Runner
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   Sequentially executes all preprocessing, gap-filling, attribution modeling,
%   and spatial analysis steps locally for the PIXEL-LEVEL (Grid-wise) approach.
% =========================================================================

clear; clc;
t_start = tic;
fprintf('=================================================================\n');
fprintf('  GLOBAL TWS ATTRIBUTION & TREND PIPELINE - GRID-WISE MLR       \n');
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

%% STEP 1: Unit Standardization & Data Sanitization
fprintf('--> Running STEP 1: Hydroclimate Flux Standardization...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'preprocessing', 'step01_unit_conversion.m'));
fprintf('✓ STEP 1 Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 2b: Grid-wise Basin Extraction
fprintf('--> Running STEP 2b: Grid-wise Extraction across 103 Basins...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'preprocessing', 'step02b_extract_gridwise.m'));
fprintf('✓ STEP 2b Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 3b: ML (RF) GRACE Gap-Filling
fprintf('--> Running STEP 3b: ML (RF) GRACE Gap-filling & Reconstruction...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'gap_filling', 'step03b_reconstruct_gridwise.m'));
fprintf('✓ STEP 3b Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 4c: Twin MLR Machine Learning Attribution
fprintf('--> Running STEP 4c: Twin MLR Attribution Modeling (Anomalies)...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'modeling', 'step04c_run_attribution_gridwise_mlr.m'));
fprintf('✓ STEP 4c Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 5b: Basin Spatial Variability Analysis
fprintf('--> Running STEP 5b: Spatial Variability & Overall Equations...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'validation', 'step05b_analyze_spatial_variability.m'));
fprintf('✓ STEP 5b Completed in %.2f seconds.\n\n', toc(t_step));

%% Summary
fprintf('=================================================================\n');
fprintf('  GRID-WISE PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN %.2f SECS\n', toc(t_start));
fprintf('  Outputs saved to: %s\n', fullfile(project_root, 'outputs'));
fprintf('=================================================================\n');
