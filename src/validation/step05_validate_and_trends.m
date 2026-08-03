%% step05_validate_and_trends.m - Block CV Validation, Hydrologic Metrics & Modified Mann-Kendall Trends
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Rigorous Cross-Validation (GEMINI.md Directive): Perform 3-Year 
%      Contiguous Block Cross-Validation (36 consecutive months test blocks)
%      to prevent temporal autocorrelation leakage.
%   2. Compute Hydrologic Metrics per basin for both models:
%      - Nash-Sutcliffe Efficiency (NSE)
%      - Kling-Gupta Efficiency (KGE)
%      - Root Mean Square Error (RMSE)
%      - Pearson Coefficient of Determination (R^2)
%   3. Estimate TWS Trend Magnitude: Theil-Sen's Slope Estimator (cm/year).
%   4. Assess Trend Significance: Hamed & Rao (1998) Modified Mann-Kendall 
%      Test (autocorrelation-corrected at p < 0.05).
%
% OUTPUT:
%   - outputs/tables/validation_and_trends.mat
%   - outputs/tables/basin_summary_table.csv
% =========================================================================

clear; clc;
fprintf('=== STEP 5: Starting Block CV Validation & Modified Mann-Kendall Trend Analysis ===\n');

%% Directory Setup & Data Loading
processed_dir = fullfile('data', 'processed');
attr_mat      = fullfile('outputs', 'tables', 'attribution_results.mat');
tws_mat       = fullfile(processed_dir, 'grace_reconstructed.mat');
ts_mat        = fullfile(processed_dir, 'basin_time_series.mat');
table_dir     = fullfile('outputs', 'tables');

if ~exist(attr_mat, 'file')
    error('Attribution results file %s not found! Run step04_run_attribution.m first.', attr_mat);
end

fprintf('Loading attribution results and continuous TWS data...\n');
attr_data = load(attr_mat);
tws_data  = load(tws_mat);
ts_data   = load(ts_mat);

TWS         = tws_data.TWS_reconstructed; % N_time x 103 continuous TWS (cm)
TWSC_obs    = attr_data.TWSC_obs;          % N_time x 103 (cm/month)
P_basin     = ts_data.P_basin;
ET_basin    = ts_data.ET_basin;
Q_basin     = ts_data.Q_basin;
GW_basin    = ts_data.GW_basin;
SW_basin    = ts_data.SW_basin;
grace_dates = tws_data.grace_dates;

[n_time, n_basins] = size(TWS);
fprintf('Datasets loaded: %d months | %d basins.\n', n_time, n_basins);

%% =========================================================================
%% 1. 3-Year Contiguous Block Cross-Validation
%% =========================================================================
fprintf('\n--- Executing 3-Year Contiguous Block Cross-Validation ---\n');

% Block size = 36 consecutive months (3 years)
block_len = 36;
n_blocks  = floor(n_time / block_len);

% Preallocate Block CV predictions
TWSC_cv_nat    = nan(n_time, n_basins);
TWSC_cv_anthro = nan(n_time, n_basins);

% Metrics containers
NSE_nat     = nan(1, n_basins);
KGE_nat     = nan(1, n_basins);
RMSE_cv_nat = nan(1, n_basins);

NSE_anthro     = nan(1, n_basins);
KGE_anthro     = nan(1, n_basins);
RMSE_cv_anthro = nan(1, n_basins);

num_cores = feature('numcores');
if isempty(gcp('nocreate'))
    parpool('local', num_cores);
end

n_trees = 150;

parfor b = 1:n_basins
    y = TWSC_obs(:, b);
    p = P_basin(:, b);
    e = ET_basin(:, b);
    
    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q = Q_basin(:, b);
    else
        q = zeros(n_time, 1);
    end
    
    if ~isempty(GW_basin) && ~all(isnan(GW_basin(:, b)))
        gw = GW_basin(:, b);
    else
        gw = zeros(n_time, 1);
    end
    
    if ~isempty(SW_basin) && ~all(isnan(SW_basin(:, b)))
        sw = SW_basin(:, b);
    else
        sw = zeros(n_time, 1);
    end
    
    X_nat = [p, e, q];
    X_ant = [p, e, q, gw, sw];
    
    y_pred_nat_cv = nan(n_time, 1);
    y_pred_ant_cv = nan(n_time, 1);
    
    % Block Cross-Validation Loop
    for k = 1:n_blocks
        test_idx  = false(n_time, 1);
        start_idx = (k - 1) * block_len + 1;
        end_idx   = min(k * block_len, n_time);
        test_idx(start_idx:end_idx) = true;
        
        train_idx = ~test_idx;
        
        % Filter valid observations for training
        valid_train = train_idx & ~isnan(y) & ~any(isnan(X_ant), 2);
        valid_test  = test_idx & ~isnan(y) & ~any(isnan(X_ant), 2);
        
        if sum(valid_train) > 20 && any(valid_test)
            % Model 1: Natural Baseline
            rf_nat_k = TreeBagger(n_trees, X_nat(valid_train, :), y(valid_train), 'Method', 'regression');
            y_pred_nat_cv(valid_test) = predict(rf_nat_k, X_nat(valid_test, :));
            
            % Model 2: Anthropogenic
            rf_ant_k = TreeBagger(n_trees, X_ant(valid_train, :), y(valid_train), 'Method', 'regression');
            y_pred_ant_cv(valid_test) = predict(rf_ant_k, X_ant(valid_test, :));
        end
    end
    
    TWSC_cv_nat(:, b)    = y_pred_nat_cv;
    TWSC_cv_anthro(:, b) = y_pred_ant_cv;
    
    % Compute Hydrologic Metrics for Basin b
    valid_cv = ~isnan(y) & ~isnan(y_pred_nat_cv) & ~isnan(y_pred_ant_cv);
    if sum(valid_cv) > 10
        y_v     = y(valid_cv);
        y_nat_v = y_pred_nat_cv(valid_cv);
        y_ant_v = y_pred_ant_cv(valid_cv);
        
        % Natural Model Metrics
        NSE_nat(b)     = compute_nse(y_v, y_nat_v);
        KGE_nat(b)     = compute_kge(y_v, y_nat_v);
        RMSE_cv_nat(b) = sqrt(mean((y_v - y_nat_v).^2));
        
        % Anthropogenic Model Metrics
        NSE_anthro(b)     = compute_nse(y_v, y_ant_v);
        KGE_anthro(b)     = compute_kge(y_v, y_ant_v);
        RMSE_cv_anthro(b) = sqrt(mean((y_v - y_ant_v).^2));
    end
end

fprintf('Block CV complete.\n');
fprintf('Mean Natural Model CV  -> NSE: %.3f | KGE: %.3f | RMSE: %.3f cm/mo\n', ...
    mean(NSE_nat, 'omitnan'), mean(KGE_nat, 'omitnan'), mean(RMSE_cv_nat, 'omitnan'));
fprintf('Mean Anthro Model CV   -> NSE: %.3f | KGE: %.3f | RMSE: %.3f cm/mo\n', ...
    mean(NSE_anthro, 'omitnan'), mean(KGE_anthro, 'omitnan'), mean(RMSE_cv_anthro, 'omitnan'));

%% =========================================================================
%% 2. Theil-Sen Trend Slope & Hamed-Rao Modified Mann-Kendall Test
%% =========================================================================
fprintf('\n--- Computing Long-Term TWS Decline Trends & Significance ---\n');

tws_trend_slope  = nan(1, n_basins); % cm/year
mk_p_value       = nan(1, n_basins); % Autocorrelation-corrected p-value
mk_h_sig         = false(1, n_basins); % True if p < 0.05

time_years = (1:n_time)' / 12.0; % Convert months to fractional years

for b = 1:n_basins
    tws_b = TWS(:, b);
    valid_tws = ~isnan(tws_b);
    
    if sum(valid_tws) > 30
        y_tws = tws_b(valid_tws);
        x_tws = time_years(valid_tws);
        
        % Compute Theil-Sen's Slope Estimator (cm/year)
        tws_trend_slope(b) = theil_sen_slope(x_tws, y_tws);
        
        % Compute Hamed & Rao (1998) Modified Mann-Kendall Test
        [p_val, h_sig] = hamed_rao_mann_kendall(y_tws, 0.05);
        mk_p_value(b) = p_val;
        mk_h_sig(b)   = h_sig;
    end
end

fprintf('Trend Analysis Complete.\n');
fprintf('Significant TWS Decline Basins (p < 0.05): %d / %d\n', sum(mk_h_sig & tws_trend_slope < 0), n_basins);
fprintf('Mean TWS Trend Rate across declining basins: %.2f cm/year\n', ...
    mean(tws_trend_slope(mk_h_sig & tws_trend_slope < 0), 'omitnan'));

%% =========================================================================
%% 3. Save Summary Outputs & CSV Report Table
%% =========================================================================
out_mat = fullfile(table_dir, 'validation_and_trends.mat');
save(out_mat, 'NSE_nat', 'NSE_anthro', 'KGE_nat', 'KGE_anthro', ...
              'RMSE_cv_nat', 'RMSE_cv_anthro', 'TWSC_cv_nat', 'TWSC_cv_anthro', ...
              'tws_trend_slope', 'mk_p_value', 'mk_h_sig', '-v7.3');

% Export Publication-Grade CSV Summary Table
csv_file = fullfile(table_dir, 'basin_summary_table.csv');
Basin_ID = (1:n_basins)';
Trend_cm_yr = tws_trend_slope';
MK_p_value  = mk_p_value';
Is_Significant = mk_h_sig';
NSE_Natural = NSE_nat';
NSE_Anthro  = NSE_anthro';
KGE_Natural = KGE_nat';
KGE_Anthro  = KGE_anthro';

summary_tbl = table(Basin_ID, Trend_cm_yr, MK_p_value, Is_Significant, ...
                    NSE_Natural, NSE_Anthro, KGE_Natural, KGE_Anthro);
writetable(summary_tbl, csv_file);
fprintf('Exported summary table to %s\n', csv_file);
fprintf('=== STEP 5 Complete: Validation & Trend Analysis Saved ===\n\n');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function nse = compute_nse(obs, sim)
    % Nash-Sutcliffe Efficiency (NSE)
    numerator   = sum((obs - sim).^2);
    denominator = sum((obs - mean(obs)).^2);
    nse = 1 - (numerator / denominator);
end

function kge = compute_kge(obs, sim)
    % Kling-Gupta Efficiency (KGE)
    r     = corr(obs, sim, 'rows', 'complete');
    alpha = std(sim) / std(obs);
    beta  = mean(sim) / mean(obs);
    kge   = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2);
end

function slope = theil_sen_slope(x, y)
    % Theil-Sen's robust non-parametric slope estimator
    n = length(y);
    slopes = [];
    count = 1;
    for i = 1:n-1
        for j = i+1:n
            if x(j) ~= x(i)
                slopes(count) = (y(j) - y(i)) / (x(j) - x(i));
                count = count + 1;
            end
        end
    end
    slope = median(slopes);
end

function [p_val, h_sig] = hamed_rao_mann_kendall(y, alpha_sig)
    % Hamed & Rao (1998) Modified Mann-Kendall Test for Autocorrelated Data
    n = length(y);
    
    % Standard Mann-Kendall S statistic
    S = 0;
    for k = 1:n-1
        for j = k+1:n
            S = S + sign(y(j) - y(k));
        end
    end
    
    % Compute autocorrelation on detrended series
    x = (1:n)';
    b = theil_sen_slope(x, y);
    detrended = y - b * x;
    
    % Autocorrelation up to lag n/4
    max_lag = floor(n / 4);
    autocorr_vec = zeros(max_lag, 1);
    var_d = var(detrended);
    for lag = 1:max_lag
        c_lag = cov(detrended(1:end-lag), detrended(1+lag:end));
        if var_d > 0
            autocorr_vec(lag) = c_lag(1, 2) / var_d;
        end
    end
    
    % Variance correction factor (Hamed & Rao 1998)
    n_star = 0;
    for i = 1:max_lag
        if abs(autocorr_vec(i)) > 1.96 / sqrt(n)
            n_star = n_star + (n - i) * (n - i - 1) * (n - i - 2) * autocorr_vec(i);
        end
    end
    
    var_S0 = n * (n - 1) * (2*n + 5) / 18;
    var_S  = var_S0 * (1 + (2 / (n * (n-1) * (n-2))) * n_star);
    
    % Z statistic computation
    if S > 0
        Z = (S - 1) / sqrt(var_S);
    elseif S < 0
        Z = (S + 1) / sqrt(var_S);
    else
        Z = 0;
    end
    
    p_val = 2 * (1 - normcdf(abs(Z)));
    h_sig = (p_val < alpha_sig);
end
