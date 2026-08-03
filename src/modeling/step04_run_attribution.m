%% step04_run_attribution.m - Twin Random Forest Machine Learning Driver Attribution
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% GOVERNING MASS BALANCE: dTWS/dt = TWSC = P - ET - Q - (GW_abs + SW_abs)
% =========================================================================
% PURPOSE:
%   1. Compute Terrestrial Water Storage Change (TWSC) using centered finite differences:
%        TWSC(t) = [TWS(t+1) - TWS(t-1)] / (2 * dt)
%   2. Train Twin Random Forest Models per basin in parallel (parfor):
%      - Model 1 (Natural Baseline M_nat):     TWSC = f(P, ET, Q)
%      - Model 2 (Full Anthropogenic M_anthro): TWSC = f(P, ET, Q, GW_abs, SW_abs)
%   3. Compute Attribution Metrics:
%      - Variance Explained Gain: Delta R^2 = R^2_anthro - R^2_nat
%      - Out-of-Bag (OOB) Feature Permutation Importance per hydroclimate driver
%
% OUTPUT:
%   - outputs/tables/attribution_results.mat containing:
%     R2_nat, R2_anthro, Delta_R2, feature_importance, TWSC_obs, TWSC_pred_nat, TWSC_pred_anthro
% =========================================================================

clear; clc;
fprintf('=== STEP 4: Starting Twin RF Machine Learning Attribution ===\n');

%% Directory Setup & Data Loading
processed_dir = fullfile('data', 'processed');
output_dir    = fullfile('outputs', 'tables');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

tws_file  = fullfile(processed_dir, 'grace_reconstructed.mat');
ts_file   = fullfile(processed_dir, 'basin_time_series.mat');

if ~exist(tws_file, 'file')
    error('Reconstructed TWS file %s not found! Run step03_reconstruct_grace.m first.', tws_file);
end

fprintf('Loading reconstructed TWS and hydroclimate predictors...\n');
tws_data = load(tws_file);
ts_data  = load(ts_file);

TWS         = tws_data.TWS_reconstructed; % N_time x 103 continuous cm depth
P_basin     = ts_data.P_basin;             % cm/month
ET_basin    = ts_data.ET_basin;            % cm/month
Q_basin     = ts_data.Q_basin;             % cm/month
GW_basin    = ts_data.GW_basin;            % cm/month
SW_basin    = ts_data.SW_basin;            % cm/month
grace_dates = tws_data.grace_dates;

[n_time, n_basins] = size(TWS);
fprintf('Time points: %d | Basins: %d\n', n_time, n_basins);

%% 1. Compute TWSC via Centered Finite Differences
% -------------------------------------------------------------------------
% TWSC(t) = [TWS(t+1) - TWS(t-1)] / (2 * dt)
% Units: cm/month (with dt = 1 month)
fprintf('Calculating TWSC via centered finite differences...\n');
TWSC_obs = nan(n_time, n_basins);

dt = 1.0; % Monthly time-step unit
for b = 1:n_basins
    tws_b = TWS(:, b);
    
    % Centered finite differences for interior points (t = 2..N-1)
    TWSC_obs(2:end-1, b) = (tws_b(3:end) - tws_b(1:end-2)) / (2 * dt);
    
    % Forward difference for start point (t = 1)
    TWSC_obs(1, b) = (tws_b(2) - tws_b(1)) / dt;
    
    % Backward difference for end point (t = N)
    TWSC_obs(end, b) = (tws_b(end) - tws_b(end-1)) / dt;
end

%% Initialize Result Matrices
R2_nat           = nan(1, n_basins);
R2_anthro        = nan(1, n_basins);
Delta_R2         = nan(1, n_basins);
RMSE_nat         = nan(1, n_basins);
RMSE_anthro      = nan(1, n_basins);

TWSC_pred_nat    = nan(n_time, n_basins);
TWSC_pred_anthro = nan(n_time, n_basins);

% Feature Importance matrix: [103 basins x 5 features]
% Features: 1:P, 2:ET, 3:Q, 4:GW_abs, 5:SW_abs
feature_importance = nan(n_basins, 5);

%% Parallel Processing Setup
num_cores = feature('numcores');
if isempty(gcp('nocreate'))
    parpool('local', num_cores);
end

n_trees       = 200;
min_leaf_size = 5;

fprintf('\nRunning Twin RF Attribution Models in parallel across %d basins...\n', n_basins);

%% Parfor Parallel Attribution Execution
parfor b = 1:n_basins
    twsc_target = TWSC_obs(:, b);
    p_b  = P_basin(:, b);
    et_b = ET_basin(:, b);
    
    % Runoff handling
    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q_b = Q_basin(:, b);
    else
        q_b = zeros(n_time, 1);
    end
    
    % Abstraction handling
    if ~isempty(GW_basin) && ~all(isnan(GW_basin(:, b)))
        gw_b = GW_basin(:, b);
    else
        gw_b = zeros(n_time, 1);
    end
    
    if ~isempty(SW_basin) && ~all(isnan(SW_basin(:, b)))
        sw_b = SW_basin(:, b);
    else
        sw_b = zeros(n_time, 1);
    end
    
    % Temporary output vectors for this basin
    twsc_nat_b    = nan(n_time, 1);
    twsc_anthro_b = nan(n_time, 1);
    
    % Feature matrices
    % Model 1 (Natural): P, ET, Q
    X_nat = [p_b, et_b, q_b];
    
    % Model 2 (Anthropogenic): P, ET, Q, GW_abs, SW_abs
    X_anthro = [p_b, et_b, q_b, gw_b, sw_b];
    
    % Valid sample mask
    valid_mask = ~isnan(twsc_target) & ~any(isnan(X_anthro), 2);
    if sum(valid_mask) < 30
        continue;
    end
    
    y_val      = twsc_target(valid_mask);
    X_nat_val  = X_nat(valid_mask, :);
    X_ant_val  = X_anthro(valid_mask, :);
    
    %% --- Train Model 1 (M_nat) ---
    rf_nat = TreeBagger(n_trees, X_nat_val, y_val, ...
        'Method', 'regression', ...
        'OOBPrediction', 'on', ...
        'MinLeafSize', min_leaf_size);
    
    y_oob_nat = oobPredict(rf_nat);
    res_nat   = y_val - y_oob_nat;
    ss_tot    = sum((y_val - mean(y_val)).^2);
    
    R2_nat(b)   = 1 - (sum(res_nat.^2) / ss_tot);
    RMSE_nat(b) = sqrt(mean(res_nat.^2));
    
    twsc_nat_b(valid_mask) = y_oob_nat;
    TWSC_pred_nat(:, b)    = twsc_nat_b;
    
    %% --- Train Model 2 (M_anthro) ---
    rf_anthro = TreeBagger(n_trees, X_ant_val, y_val, ...
        'Method', 'regression', ...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on', ...
        'MinLeafSize', min_leaf_size);
    
    y_oob_ant = oobPredict(rf_anthro);
    res_ant   = y_val - y_oob_ant;
    
    R2_anthro(b)   = 1 - (sum(res_ant.^2) / ss_tot);
    RMSE_anthro(b) = sqrt(mean(res_ant.^2));
    
    twsc_anthro_b(valid_mask) = y_oob_ant;
    TWSC_pred_anthro(:, b)    = twsc_anthro_b;
    
    % Variance Explained Gain: Delta R^2
    Delta_R2(b) = R2_anthro(b) - R2_nat(b);
    
    % Feature Permutation Importance Scores
    feature_importance(b, :) = rf_anthro.OOBPermutedPredictorDeltaError;
end

fprintf('\nAttribution modeling complete.\n');
fprintf('Average R^2 (Natural Model M_nat):     %.3f\n', mean(R2_nat, 'omitnan'));
fprintf('Average R^2 (Anthropogenic M_anthro):  %.3f\n', mean(R2_anthro, 'omitnan'));
fprintf('Average Delta R^2 (Anthropogenic Gain): %.3f\n', mean(Delta_R2, 'omitnan'));

%% Save Attribution Output
out_mat = fullfile(output_dir, 'attribution_results.mat');
fprintf('Saving attribution results matrix to %s...\n', out_mat);
save(out_mat, 'TWSC_obs', 'TWSC_pred_nat', 'TWSC_pred_anthro', ...
              'R2_nat', 'R2_anthro', 'Delta_R2', 'RMSE_nat', 'RMSE_anthro', ...
              'feature_importance', 'grace_dates', '-v7.3');
fprintf('=== STEP 4 Complete: Twin Model Attribution Saved ===\n\n');
