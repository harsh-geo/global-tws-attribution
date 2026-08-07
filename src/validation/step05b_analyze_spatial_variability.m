%% step05b_analyze_spatial_variability.m - Analyze MLR Spatial Variability
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Load grid-wise attribution results (betas and R2 values).
%   2. For each basin, derive the "Overall Basin Equation" by taking the 
%      spatial mean of the multiple linear regression coefficients.
%   3. Calculate spatial variability metrics (standard deviation) for
%      Delta R^2 and the MLR coefficients to map internal basin dynamics.
%
% OUTPUT:
%   - outputs/tables/basin_overall_equations.csv
%   - outputs/tables/spatial_variability_metrics.mat
% =========================================================================

fprintf('=== STEP 5b: Starting Spatial Variability Analysis ===\n');

%% Directory Setup & Data Loading
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
if isempty(project_root) || ~exist(fullfile(project_root, 'outputs'), 'dir')
    project_root = pwd;
end

output_dir = fullfile(project_root, 'outputs', 'tables');
attr_file  = fullfile(output_dir, 'attribution_results_gridwise.mat');

if exist(attr_file, 'file')
    fprintf('Loading grid-wise attribution results...\n');
    load(attr_file);
else
    fprintf('Attribution results not found. Run step04c first.\n');
    return;
end

n_basins = length(beta_nat_gridwise);

%% Initialize Metrics
% Basin Overall Equation (Mean Coefficients)
mean_beta_nat    = nan(n_basins, 4); % [Intercept, P, ET, Q]
mean_beta_anthro = nan(n_basins, 6); % [Intercept, P, ET, Q, GW, SW]
mean_delta_r2    = nan(n_basins, 1);

% Spatial Variability (Standard Deviation)
std_beta_nat     = nan(n_basins, 4);
std_beta_anthro  = nan(n_basins, 6);
std_delta_r2     = nan(n_basins, 1);

%% Calculation Loop
for b = 1:n_basins
    b_nat = beta_nat_gridwise{b};    % 4 x N_pixels
    b_ant = beta_anthro_gridwise{b}; % 6 x N_pixels
    dr2   = Delta_R2_gridwise{b};    % 1 x N_pixels
    
    if isempty(b_nat) || all(isnan(b_nat(:)))
        continue;
    end
    
    % Mean metrics (Overall Equation)
    mean_beta_nat(b, :)    = mean(b_nat, 2, 'omitnan')';
    mean_beta_anthro(b, :) = mean(b_ant, 2, 'omitnan')';
    mean_delta_r2(b)       = mean(dr2, 2, 'omitnan');
    
    % Spatial Variability metrics (Standard Deviation)
    std_beta_nat(b, :)    = std(b_nat, 0, 2, 'omitnan')';
    std_beta_anthro(b, :) = std(b_ant, 0, 2, 'omitnan')';
    std_delta_r2(b)       = std(dr2, 0, 2, 'omitnan');
end

%% Write Overall Equations to CSV for Easy Reading
basin_ids = (1:n_basins)';

T_nat = table(basin_ids, mean_beta_nat(:,1), mean_beta_nat(:,2), mean_beta_nat(:,3), mean_beta_nat(:,4), ...
    'VariableNames', {'BasinID', 'Intercept', 'Beta_P', 'Beta_ET', 'Beta_Q'});

T_ant = table(basin_ids, mean_beta_anthro(:,1), mean_beta_anthro(:,2), mean_beta_anthro(:,3), ...
    mean_beta_anthro(:,4), mean_beta_anthro(:,5), mean_beta_anthro(:,6), mean_delta_r2, ...
    'VariableNames', {'BasinID', 'Intercept', 'Beta_P', 'Beta_ET', 'Beta_Q', 'Beta_GW', 'Beta_SW', 'Mean_Delta_R2'});

csv_nat = fullfile(output_dir, 'basin_overall_equations_natural.csv');
csv_ant = fullfile(output_dir, 'basin_overall_equations_anthropogenic.csv');

writetable(T_nat, csv_nat);
writetable(T_ant, csv_ant);

fprintf('Saved overall basin equations to CSVs in outputs/tables/.\n');

%% Save Variability Metrics
var_file = fullfile(output_dir, 'spatial_variability_metrics.mat');
save(var_file, 'mean_beta_nat', 'mean_beta_anthro', 'mean_delta_r2', ...
    'std_beta_nat', 'std_beta_anthro', 'std_delta_r2');
fprintf('Saved spatial variability metrics to %s.\n', var_file);

fprintf('=== STEP 5b Complete ===\n\n');
