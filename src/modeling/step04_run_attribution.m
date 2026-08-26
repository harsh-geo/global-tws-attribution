%% step04_run_attribution.m - Twin RF Attribution (Baseline Anomaly Framework)
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% GOVERNING MASS BALANCE: dTWS/dt = TWSC = P - ET - Q - (GW_abs + SW_abs)
% =========================================================================
% PURPOSE:
%   1. Load gap-filled TWS and basin hydroclimate predictors.
%   2. Calculate TWSC using central finite differences.
%   3. Deseasonalize all variables using the 2004-2009 GRACE baseline.
%   4. Train Twin Random Forest Models per basin in parallel (parfor):
%      - Model 1 (Natural Baseline M_nat):     TWSC = f(P_anom, ET_anom, Q_anom)
%      - Model 2 (Full Anthropogenic M_anthro): TWSC = f(P_anom, ET_anom, Q_anom, GW_anom, SW_anom)
%   5. Compute Attribution Metrics:
%      - Variance Explained Gain: Delta R^2 = R^2_anthro - R^2_nat
%      - Out-of-Bag (OOB) Feature Permutation Importance per hydroclimate driver
%
% OUTPUT:
%   - outputs/tables/attribution_results.mat containing:
%     R2_nat, R2_anthro, Delta_R2, feature_importance, TWSC_obs, TWSC_pred_nat, TWSC_pred_anthro
% =========================================================================

fprintf('=== STEP 4: Starting Twin RF Machine Learning Attribution (Anomalies) ===\n');

%% Directory Setup & Data Loading
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

processed_dir = fullfile(project_root, 'data', 'processed');
output_dir    = fullfile(project_root, 'outputs', 'tables');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

% Check if inputs are in memory
if ~exist('TWS_reconstructed', 'var') || isempty(TWS_reconstructed)
    tws_file = fullfile(processed_dir, 'grace_reconstructed.mat');
    if exist(tws_file, 'file')
        fprintf('Loading reconstructed TWS from %s...\n', tws_file);
        tws_data = load(tws_file);
        TWS_reconstructed = tws_data.TWS_reconstructed;
    else
        fprintf('Reconstructed TWS not in memory. Running step03_reconstruct_grace.m...\n');
        run(fullfile(project_root, 'src', 'gap_filling', 'step03_reconstruct_grace.m'));
    end
end
TWS = TWS_reconstructed;

if ~exist('P_basin', 'var') || isempty(P_basin) || ...
        ~exist('ET_basin', 'var') || isempty(ET_basin) || ...
        ~exist('GW_basin', 'var') || isempty(GW_basin) || ...
        ~exist('SW_basin', 'var') || isempty(SW_basin)
    ts_file = fullfile(processed_dir, 'basin_time_series.mat');
    if exist(ts_file, 'file')
        fprintf('Loading basin time-series from %s...\n', ts_file);
        ts_data = load(ts_file);
        P_basin     = ts_data.P_basin;
        ET_basin    = ts_data.ET_basin;
        Q_basin     = ts_data.Q_basin;
        GW_basin    = ts_data.GW_basin;
        SW_basin    = ts_data.SW_basin;
        grace_dates = ts_data.grace_dates;
    else
        fprintf('Basin time-series not in memory. Running step02_aggregate_basins.m...\n');
        run(fullfile(project_root, 'src', 'preprocessing', 'step02_aggregate_basins.m'));
    end
end

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

n_trees       = 500;
min_leaf_size = 5;

% Tuning variables to sample to force selection of weak anthropogenic signals
n_vars_sample_nat = 1;  % M_nat has 3 predictors (P_anom, ET_anom, Q_anom)
n_vars_sample_ant = 1;  % M_anthro has 5 predictors (P_anom, ET_anom, Q_anom, GW_anom, SW_anom)

% Define baseline indices (2004-2009) for anomaly calculation
target_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';
baseline_idx = year(target_dates) >= 2004 & year(target_dates) <= 2009;

fprintf('\nRunning Twin RF Attribution Models (Anomalies 2004-2009 baseline) in parallel across %d basins...\n', n_basins);

%% Parfor Parallel Attribution Execution
parfor b = 1:n_basins
    twsc_target = TWSC_obs(:, b);
    twsc_target = deseasonalize(twsc_target, target_dates, baseline_idx);

    p_b  = P_basin(:, b);
    p_b  = deseasonalize(p_b, target_dates, baseline_idx); % Deseasonalised Anomaly (2004-2009 baseline)

    et_b = ET_basin(:, b);
    et_b = deseasonalize(et_b, target_dates, baseline_idx); % Deseasonalised Anomaly (2004-2009 baseline)

    % Runoff handling
    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q_b = Q_basin(:, b);
        q_b = deseasonalize(q_b, target_dates, baseline_idx); % Deseasonalised Anomaly (2004-2009 baseline)
    else
        q_b = zeros(n_time, 1);
    end

    % Abstraction handling
    if ~isempty(GW_basin) && ~all(isnan(GW_basin(:, b)))
        gw_b = GW_basin(:, b);
        gw_b = deseasonalize(gw_b, target_dates, baseline_idx); % Deseasonalised Anomaly (2004-2009 baseline)
    else
        gw_b = zeros(n_time, 1);
    end

    if ~isempty(SW_basin) && ~all(isnan(SW_basin(:, b)))
        sw_b = SW_basin(:, b);
        sw_b = deseasonalize(sw_b, target_dates, baseline_idx); % Deseasonalised Anomaly (2004-2009 baseline)
    else
        sw_b = zeros(n_time, 1);
    end

    % Temporary output vectors for this basin
    twsc_nat_b    = nan(n_time, 1);
    twsc_anthro_b = nan(n_time, 1);

    % Feature matrices
    % Model 1 (Natural): P_anom, ET_anom, Q_anom
    X_nat = [p_b, et_b, q_b];

    % Model 2 (Anthropogenic): P_anom, ET_anom, Q_anom, GW_abs_anom, SW_abs_anom
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
        'MinLeafSize', min_leaf_size, ...
        'NumPredictorsToSample', n_vars_sample_nat);

    y_oob_nat = oobPredict(rf_nat);
    res_nat   = y_val - y_oob_nat;
    ss_tot    = sum((y_val - mean(y_val, 'omitnan')).^2, 'omitnan');

    R2_nat(b)   = 1 - (sum(res_nat.^2, 'omitnan') / ss_tot);
    RMSE_nat(b) = sqrt(mean(res_nat.^2, 'omitnan'));

    twsc_nat_b(valid_mask) = y_oob_nat;
    TWSC_pred_nat(:, b)    = twsc_nat_b;

    %% --- Train Model 2 (M_anthro) ---
    rf_anthro = TreeBagger(n_trees, X_ant_val, y_val, ...
        'Method', 'regression', ...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on', ...
        'MinLeafSize', min_leaf_size, ...
        'NumPredictorsToSample', n_vars_sample_ant);

    y_oob_ant = oobPredict(rf_anthro);
    res_ant   = y_val - y_oob_ant;

    R2_anthro(b)   = 1 - (sum(res_ant.^2, 'omitnan') / ss_tot);
    RMSE_anthro(b) = sqrt(mean(res_ant.^2, 'omitnan'));

    twsc_anthro_b(valid_mask) = y_oob_ant;
    TWSC_pred_anthro(:, b)    = twsc_anthro_b;

    % Variance Explained Gain: Delta R^2
    Delta_R2(b) = R2_anthro(b) - R2_nat(b);

    % Feature Permutation Importance Scores
    feature_importance(b, :) = rf_anthro.OOBPermutedPredictorDeltaError;
end

fprintf('\nAttribution modeling complete (Anomalies).\n');
fprintf('Average R^2 (Natural Model M_nat):     %.3f\n', mean(R2_nat, 'omitnan'));
fprintf('Average R^2 (Anthropogenic M_anthro):  %.3f\n', mean(R2_anthro, 'omitnan'));
fprintf('Average Delta R^2 (Anthropogenic Gain): %.3f\n', mean(Delta_R2, 'omitnan'));

%% Save Attribution Results to Disk
attr_file = fullfile(output_dir, 'attribution_results.mat');
fprintf('Saving attribution results to %s...\n', attr_file);
save(attr_file, 'R2_nat', 'R2_anthro', 'Delta_R2', 'RMSE_nat', 'RMSE_anthro', ...
    'feature_importance', 'TWSC_obs', 'TWSC_pred_nat', 'TWSC_pred_anthro', '-v7.3');
fprintf('=== STEP 4 Complete: Twin Model Attribution (Anomalies) Completed & Saved ===\n\n');

%% Helper Function for Deseasonalization with Baseline
function x_deseason = deseasonalize(x, dates, baseline_idx)
x_deseason = nan(size(x));
valid_idx = ~isnan(x);

if sum(valid_idx) < 24 % STL requires sufficient data points
    months = month(dates);
    for m = 1:12
        idx_m = (months == m);
        idx_base = idx_m & baseline_idx;
        if any(idx_base) && ~all(isnan(x(idx_base)))
            monthly_mean = mean(x(idx_base), 'omitnan');
        else
            monthly_mean = mean(x(idx_m), 'omitnan'); % fallback
        end
        x_deseason(idx_m) = x(idx_m) - monthly_mean;
    end
    return;
end

% Interpolate NaNs temporarily for STL
x_filled = x;
if any(~valid_idx)
    x_filled = fillmissing(x, 'linear');
end

% Perform STL decomposition (Period = 12 for monthly data)
[LT, ST, R] = trenddecomp(x_filled, 'stl', 12);

% Deseasonalized data is Trend (LT) + Residual (R)
x_deseas_temp = LT + R;

% Center the anomaly relative to the 2004-2009 baseline period
baseline_mean = mean(x_deseas_temp(baseline_idx), 'omitnan');
x_deseason = x_deseas_temp - baseline_mean;

% Restore NaNs from original data
    x_deseason(~valid_idx) = NaN;
end
