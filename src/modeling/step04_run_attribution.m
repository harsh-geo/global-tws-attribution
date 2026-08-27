%% step04_run_attribution.m - Unified Twin RF Attribution & SHAP Uncertainty
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% GOVERNING MASS BALANCE: dTWS/dt = TWSC = P - ET - Q - (GW_abs + SW_abs)
% =========================================================================
% PURPOSE:
%   1. Calculate TWSC via central finite differences and deseasonalize.
%   2. Train True Twin Random Forest Models (M_nat, M_anthro) per basin.
%   3. Compute formal SHAP feature attribution on the Anthropogenic model.
%   4. Run N=100 Block-Bootstrapping to estimate 95%% Confidence Intervals
%      for both Delta R^2 and the SHAP feature importances.
%
% OUTPUT:
%   - outputs/tables/attribution_results.mat
%   - outputs/tables/attribution_uncertainty.csv
% =========================================================================

fprintf('=== STEP 4: Starting Unified RF Attribution, SHAP, and Bootstrapping ===\n');

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
end

% Check if inputs are in memory
if ~exist('TWS_reconstructed', 'var') || isempty(TWS_reconstructed)
    tws_file = fullfile(processed_dir, 'grace_reconstructed.mat');
    if exist(tws_file, 'file')
        tws_data = load(tws_file);
        TWS = tws_data.TWS_reconstructed;
    else
        error('Reconstructed TWS file not found: %s. Run step03 first.', tws_file);
    end
else
    TWS = TWS_reconstructed;
end

if ~exist('P_basin', 'var') || isempty(P_basin) || ~exist('ET_basin', 'var') || isempty(ET_basin)
    ts_file = fullfile(processed_dir, 'basin_time_series.mat');
    if exist(ts_file, 'file')
        ts_data = load(ts_file);
        P_basin     = ts_data.P_basin;
        ET_basin    = ts_data.ET_basin;
        Q_basin     = ts_data.Q_basin;
        GW_basin    = ts_data.GW_basin;
        SW_basin    = ts_data.SW_basin;
        grace_dates = ts_data.grace_dates;
    else
        error('Basin time series file not found. Run step02 first.');
    end
end

[n_time, n_basins] = size(TWS);
target_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';
baseline_idx = year(target_dates) >= 2004 & year(target_dates) <= 2009;

%% 1. Compute TWSC via Centered Finite Differences
dt = 1.0; 
TWSC_obs = nan(n_time, n_basins);
for b = 1:n_basins
    tws_b = TWS(:, b);
    TWSC_obs(2:end-1, b) = (tws_b(3:end) - tws_b(1:end-2)) / (2 * dt);
    TWSC_obs(1, b) = (tws_b(2) - tws_b(1)) / dt;
    TWSC_obs(end, b) = (tws_b(end) - tws_b(end-1)) / dt;
end

%% Hyperparameters & Storage
n_trees       = 500;
min_leaf      = 5;
n_features    = 5;
N_boot        = 100;
block_len     = 36;
num_blocks_needed = ceil(n_time / block_len);

% True Results
R2_TWS_nat          = nan(1, n_basins);
R2_TWS_anthro       = nan(1, n_basins);
Delta_R2            = nan(1, n_basins);
TWS_pred_nat_all    = nan(n_time, n_basins);
TWS_pred_anthro_all = nan(n_time, n_basins);
TWSC_pred_nat_all   = nan(n_time, n_basins);
TWSC_pred_anthro_all= nan(n_time, n_basins);

shap_results = cell(n_basins, 1);

% Bootstrapped CI Storage
delta_r2_ci_low = nan(n_basins, 1);
delta_r2_ci_upp = nan(n_basins, 1);
feat_imp_ci_low = nan(n_basins, n_features);
feat_imp_ci_upp = nan(n_basins, n_features);

boot_delta_r2_dist  = nan(N_boot, n_basins);
boot_feat_imp_dist  = nan(N_boot, n_basins, n_features);

%% Parallel Processing Setup
num_cores = feature('numcores');
if isempty(gcp('nocreate'))
    parpool('local', num_cores);
end

fprintf('\nStarting Unified Attribution for %d basins (N_boot=%d)...\n', n_basins, N_boot);

%% Parfor Loop
parfor b = 1:n_basins
    fprintf('Processing Basin %d...\n', b);
    
    [twsc_target, ST_twsc, base_twsc] = deseasonalize(TWSC_obs(:, b), target_dates, baseline_idx);
    p_b  = deseasonalize(P_basin(:, b), target_dates, baseline_idx);
    et_b = deseasonalize(ET_basin(:, b), target_dates, baseline_idx);
    
    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q_b = deseasonalize(Q_basin(:, b), target_dates, baseline_idx);
    else
        q_b = zeros(n_time, 1);
    end
    
    if ~isempty(GW_basin) && ~all(isnan(GW_basin(:, b)))
        gw_b = deseasonalize(GW_basin(:, b), target_dates, baseline_idx);
    else
        gw_b = zeros(n_time, 1);
    end
    
    if ~isempty(SW_basin) && ~all(isnan(SW_basin(:, b)))
        sw_b = deseasonalize(SW_basin(:, b), target_dates, baseline_idx);
    else
        sw_b = zeros(n_time, 1);
    end
    
    X_nat = [p_b, et_b, q_b];
    X_ant = [p_b, et_b, q_b, gw_b, sw_b];
    y_full = twsc_target;
    
    valid_mask = ~isnan(y_full) & ~any(isnan(X_ant), 2);
    if sum(valid_mask) < 36
        continue;
    end
    
    y_v = y_full(valid_mask);
    X_nat_val = X_nat(valid_mask, :);
    X_ant_val = X_ant(valid_mask, :);
    
    %% --- TRUE MODELING ---
    % Predict on ALL rows where features are available to gap-fill
    valid_pred_nat = ~any(isnan(X_nat), 2);
    valid_pred_ant = ~any(isnan(X_ant), 2);

    rf_nat = TreeBagger(n_trees, X_nat_val, y_v, 'Method', 'regression', 'MinLeafSize', min_leaf, 'NumPredictorsToSample', 1);
    rf_ant = TreeBagger(n_trees, X_ant_val, y_v, 'Method', 'regression', 'MinLeafSize', min_leaf, 'NumPredictorsToSample', 1);
    
    % Predict TWSC anomaly
    y_pred_nat = predict(rf_nat, X_nat(valid_pred_nat, :));
    y_pred_ant = predict(rf_ant, X_ant(valid_pred_ant, :));
    
    twsc_nat_anom = nan(n_time, 1); twsc_nat_anom(valid_pred_nat) = y_pred_nat;
    twsc_ant_anom = nan(n_time, 1); twsc_ant_anom(valid_pred_ant) = y_pred_ant;
    
    % Reconstruct RAW TWSC by adding back seasonality and baseline mean
    twsc_nat_raw = twsc_nat_anom + ST_twsc + base_twsc;
    twsc_ant_raw = twsc_ant_anom + ST_twsc + base_twsc;
    
    TWSC_pred_nat_all(:, b) = twsc_nat_raw;
    TWSC_pred_anthro_all(:, b) = twsc_ant_raw;
    
    % Exact Integration to TWS via Inverse Centered Finite Difference Matrix
    idx_start = find(~isnan(TWS(:, b)), 1);
    if ~isempty(idx_start)
        % Fill isolated NaNs in raw TWSC if any exist so integration doesn't fail
        twsc_nat_raw(isnan(twsc_nat_raw)) = 0; 
        twsc_ant_raw(isnan(twsc_ant_raw)) = 0;
        
        n_valid = n_time - idx_start + 1;
        A = zeros(n_valid, n_valid);
        for i = 2:n_valid-1
            A(i, i-1) = -0.5;
            A(i, i+1) =  0.5;
        end
        A(1, 1) = 1; A(1, 2) = 0; % Initial condition
        A(end, end-1) = -1; A(end, end) = 1; % Backward diff at end
        
        B_nat = twsc_nat_raw(idx_start:end); B_nat(1) = TWS(idx_start, b);
        B_ant = twsc_ant_raw(idx_start:end); B_ant(1) = TWS(idx_start, b);
        
        tws_pred_nat_sub = smoothdata(A \ B_nat, 'movmean', 3);
        tws_pred_ant_sub = smoothdata(A \ B_ant, 'movmean', 3);
        
        tws_pred_nat = nan(n_time, 1); tws_pred_nat(idx_start:end) = tws_pred_nat_sub;
        tws_pred_ant = nan(n_time, 1); tws_pred_ant(idx_start:end) = tws_pred_ant_sub;
        
        TWS_pred_nat_all(:, b) = tws_pred_nat;
        TWS_pred_anthro_all(:, b) = tws_pred_ant;
        
        valid_tws = ~isnan(TWS(:, b)) & ~isnan(tws_pred_nat) & ~isnan(tws_pred_ant);
        if sum(valid_tws) > 30
            ss_tot_tws = sum((TWS(valid_tws, b) - mean(TWS(valid_tws, b))).^2);
            R2_TWS_nat(b) = 1 - (sum((TWS(valid_tws, b) - tws_pred_nat(valid_tws)).^2) / ss_tot_tws);
            R2_TWS_anthro(b) = 1 - (sum((TWS(valid_tws, b) - tws_pred_ant(valid_tws)).^2) / ss_tot_tws);
            Delta_R2(b) = R2_TWS_anthro(b) - R2_TWS_nat(b);
        end
    end
    
    % TRUE SHAP
    predict_fcn = @(X) predict(rf_ant, X);
    try
        explainer = shapley(predict_fcn, X_ant_val, 'QueryPoints', X_ant_val);
        sv_table = explainer.Shapley;
        num_vals = double(sv_table{:, 2:end});
        
        res = struct();
        res.basin_id = b;
        res.shap_values = num_vals'; % N x features
        res.valid_mask = valid_mask;
        shap_results{b} = res;
    catch ME
        fprintf('SHAP failed for basin %d: %s\n', b, ME.message);
    end
    
    %% --- BOOTSTRAPPING FOR UNCERTAINTY ---
    max_start_idx = n_time - block_len + 1;
    b_delta_r2 = nan(N_boot, 1);
    b_feat_imp = nan(N_boot, n_features);
    
    % Fast treebagger for boot
    n_trees_boot = 50; 
    
    for iter = 1:N_boot
        block_starts = randi(max_start_idx, num_blocks_needed, 1);
        sample_indices = [];
        for k = 1:num_blocks_needed
            sample_indices = [sample_indices; (block_starts(k) : block_starts(k) + block_len - 1)']; %#ok<AGROW>
        end
        sample_indices = sample_indices(1:n_time);
        
        y_boot = y_full(sample_indices);
        X_nat_boot = X_nat(sample_indices, :);
        X_ant_boot = X_ant(sample_indices, :);
        
        boot_valid = ~isnan(y_boot) & ~any(isnan(X_ant_boot), 2);
        if sum(boot_valid) < 30
            continue;
        end
        
        y_v_boot = y_boot(boot_valid);
        X_nat_v_boot = X_nat_boot(boot_valid, :);
        X_ant_v_boot = X_ant_boot(boot_valid, :);
        
        % Train Boot Models
        rf_nat_b = TreeBagger(n_trees_boot, X_nat_v_boot, y_v_boot, 'Method', 'regression', 'MinLeafSize', min_leaf, 'NumPredictorsToSample', 1);
        rf_ant_b = TreeBagger(n_trees_boot, X_ant_v_boot, y_v_boot, 'Method', 'regression', 'MinLeafSize', min_leaf, 'NumPredictorsToSample', 1);
        
        % Predict
        y_p_nat = predict(rf_nat_b, X_nat_v_boot);
        y_p_ant = predict(rf_ant_b, X_ant_v_boot);
        
        ss_tot = sum((y_v_boot - mean(y_v_boot)).^2);
        r2_nat_i = 1 - (sum((y_v_boot - y_p_nat).^2) / ss_tot);
        r2_ant_i = 1 - (sum((y_v_boot - y_p_ant).^2) / ss_tot);
        b_delta_r2(iter) = r2_ant_i - r2_nat_i; % TWSC-based proxy for CI
        
        % Boot SHAP
        predict_fcn_b = @(X) predict(rf_ant_b, X);
        try
            % Limit SHAP query points to save extreme compute during bootstrapping
            max_q = min(size(X_ant_v_boot, 1), 60); 
            idx_q = randperm(size(X_ant_v_boot, 1), max_q);
            explainer_b = shapley(predict_fcn_b, X_ant_v_boot, 'QueryPoints', X_ant_v_boot(idx_q, :));
            sv_table_b = explainer_b.Shapley;
            num_vals_b = double(sv_table_b{:, 2:end});
            b_feat_imp(iter, :) = mean(abs(num_vals_b'), 1, 'omitnan');
        catch
        end
    end
    
    boot_delta_r2_dist(:, b) = b_delta_r2;
    boot_feat_imp_dist(:, b, :) = b_feat_imp;
    
    valid_res = ~isnan(b_delta_r2);
    if sum(valid_res) >= 5
        delta_r2_ci_low(b) = prctile(b_delta_r2(valid_res), 2.5);
        delta_r2_ci_upp(b) = prctile(b_delta_r2(valid_res), 97.5);
        for f = 1:n_features
            feat_imp_ci_low(b, f) = prctile(b_feat_imp(valid_res, f), 2.5);
            feat_imp_ci_upp(b, f) = prctile(b_feat_imp(valid_res, f), 97.5);
        end
    end
end

fprintf('\nSaving unified attribution results...\n');

%% Save MAT Results File
attr_file = fullfile(output_dir, 'attribution_results.mat');
save(attr_file, 'R2_TWS_nat', 'R2_TWS_anthro', 'Delta_R2', 'TWSC_obs', ...
    'TWSC_pred_nat_all', 'TWSC_pred_anthro_all', 'TWS_pred_nat_all', 'TWS_pred_anthro_all', ...
    'shap_results', 'boot_delta_r2_dist', 'boot_feat_imp_dist', ...
    'delta_r2_ci_low', 'delta_r2_ci_upp', 'feat_imp_ci_low', 'feat_imp_ci_upp', '-v7.3');

%% Export Summary CSV Table
out_csv = fullfile(output_dir, 'attribution_uncertainty.csv');
Basin_ID = (1:n_basins)';
Delta_R2 = Delta_R2';
Delta_R2_CI_2_5 = delta_r2_ci_low;
Delta_R2_CI_97_5 = delta_r2_ci_upp;

% Compute True Mean SHAP Importance
feat_imp = nan(n_basins, n_features);
for i = 1:n_basins
    if ~isempty(shap_results{i})
        feat_imp(i, :) = mean(abs(shap_results{i}.shap_values), 1, 'omitnan');
    end
end

Imp_P = feat_imp(:, 1); Imp_P_CI_Low = feat_imp_ci_low(:, 1); Imp_P_CI_Upp = feat_imp_ci_upp(:, 1);
Imp_ET = feat_imp(:, 2); Imp_ET_CI_Low = feat_imp_ci_low(:, 2); Imp_ET_CI_Upp = feat_imp_ci_upp(:, 2);
Imp_Q = feat_imp(:, 3); Imp_Q_CI_Low = feat_imp_ci_low(:, 3); Imp_Q_CI_Upp = feat_imp_ci_upp(:, 3);
Imp_GW = feat_imp(:, 4); Imp_GW_CI_Low = feat_imp_ci_low(:, 4); Imp_GW_CI_Upp = feat_imp_ci_upp(:, 4);
Imp_SW = feat_imp(:, 5); Imp_SW_CI_Low = feat_imp_ci_low(:, 5); Imp_SW_CI_Upp = feat_imp_ci_upp(:, 5);

T_summary = table(Basin_ID, Delta_R2, Delta_R2_CI_2_5, Delta_R2_CI_97_5, ...
    Imp_P, Imp_P_CI_Low, Imp_P_CI_Upp, ...
    Imp_ET, Imp_ET_CI_Low, Imp_ET_CI_Upp, ...
    Imp_Q, Imp_Q_CI_Low, Imp_Q_CI_Upp, ...
    Imp_GW, Imp_GW_CI_Low, Imp_GW_CI_Upp, ...
    Imp_SW, Imp_SW_CI_Low, Imp_SW_CI_Upp);
writetable(T_summary, out_csv);
fprintf('=== STEP 4 Complete ===\n');

%% Helper Deseasonalization Function with Baseline
function [x_deseason, ST, baseline_mean] = deseasonalize(x, dates, baseline_idx)
    x_deseason = nan(size(x));
    ST = zeros(size(x));
    baseline_mean = 0;
    
    valid_idx = ~isnan(x);
    if sum(valid_idx) < 24
        months = month(dates);
        for m = 1:12
            idx_m = (months == m);
            idx_base = idx_m & baseline_idx;
            if any(idx_base) && ~all(isnan(x(idx_base)))
                monthly_mean = mean(x(idx_base), 'omitnan');
            else
                monthly_mean = mean(x(idx_m), 'omitnan');
            end
            x_deseason(idx_m) = x(idx_m) - monthly_mean;
            ST(idx_m) = monthly_mean; % Approximation for short series
        end
        return;
    end
    x_filled = x;
    if any(~valid_idx)
        x_filled = fillmissing(x, 'linear');
    end
    [LT, ST_out, R] = trenddecomp(x_filled, 'stl', 12);
    x_deseas_temp = LT + R;
    baseline_mean = mean(x_deseas_temp(baseline_idx), 'omitnan');
    x_deseason = x_deseas_temp - baseline_mean;
    x_deseason(~valid_idx) = NaN;
    ST = ST_out;
end
