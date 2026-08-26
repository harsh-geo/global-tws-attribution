%% step04d_run_attribution_lstm.m - Twin LSTM Deep Learning Attribution
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% GOVERNING MASS BALANCE: dTWS/dt = TWSC = P - ET - Q - (GW_abs + SW_abs)
% =========================================================================
% PURPOSE:
%   1. Compute Terrestrial Water Storage Change (TWSC)
%   2. Train Twin LSTM Models per basin:
%      - Model 1 (Natural Baseline M_nat):     TWSC = f(P, ET, Q)
%      - Model 2 (Full Anthropogenic M_anthro): TWSC = f(P, ET, Q, GW_abs, SW_abs)
%   3. Compute Attribution Metrics:
%      - Variance Explained Gain: Delta R^2 = R^2_anthro - R^2_nat
%
% OUTPUT:
%   - outputs/tables/attribution_results_lstm.mat
% =========================================================================

fprintf('=== STEP 4d: Starting Twin LSTM Deep Learning Attribution ===\n');

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

% Load Data
if ~exist('TWS_reconstructed', 'var') || isempty(TWS_reconstructed)
    tws_file = fullfile(processed_dir, 'grace_reconstructed.mat');
    if exist(tws_file, 'file')
        tws_data = load(tws_file);
        TWS_reconstructed = tws_data.TWS_reconstructed;
    else
        run(fullfile(project_root, 'src', 'gap_filling', 'step03_reconstruct_grace.m'));
    end
end
TWS = TWS_reconstructed;

ts_file = fullfile(processed_dir, 'basin_time_series.mat');
ts_data = load(ts_file);
P_basin     = ts_data.P_basin;
ET_basin    = ts_data.ET_basin;
Q_basin     = ts_data.Q_basin;
GW_basin    = ts_data.GW_basin;
SW_basin    = ts_data.SW_basin;
grace_dates = ts_data.grace_dates;

[n_time, n_basins] = size(TWS);
fprintf('Time points: %d | Basins: %d\n', n_time, n_basins);

%% 1. Compute TWSC via Centered Finite Differences
TWSC_obs = nan(n_time, n_basins);
dt = 1.0; % Monthly time-step unit
for b = 1:n_basins
    tws_b = TWS(:, b);
    TWSC_obs(2:end-1, b) = (tws_b(3:end) - tws_b(1:end-2)) / (2 * dt);
    TWSC_obs(1, b) = (tws_b(2) - tws_b(1)) / dt;
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
TWSC_pred_anthro_upper = nan(n_time, n_basins);
TWSC_pred_anthro_lower = nan(n_time, n_basins);

feature_importance = nan(n_basins, 5);
shap_values = nan(n_time, n_basins, 5);

has_parallel = license('test', 'Distrib_Computing_Toolbox');
if has_parallel
    try
        if isempty(gcp('nocreate'))
            num_cores = feature('numcores');
            parpool('local', num_cores);
        end
    catch ME
        fprintf('Note: Running serially (Parallel pool initialization failed: %s)\n', ME.message);
    end
end

target_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';

fprintf('\nRunning Twin LSTM Attribution Models in parallel across %d basins...\n', n_basins);

% LSTM Hyperparameters
numHiddenUnits1 = 64;
numHiddenUnits2 = 32;
maxEpochs = 150;

parfor b = 1:n_basins
    twsc_b = deseasonalize(TWSC_obs(:, b), target_dates);
    p_b    = deseasonalize(P_basin(:, b), target_dates);
    et_b   = deseasonalize(ET_basin(:, b), target_dates);
    
    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q_b = deseasonalize(Q_basin(:, b), target_dates);
    else
        q_b = zeros(n_time, 1);
    end
    
    if ~isempty(GW_basin) && ~all(isnan(GW_basin(:, b)))
        gw_b = deseasonalize(GW_basin(:, b), target_dates);
    else
        gw_b = zeros(n_time, 1);
    end
    
    if ~isempty(SW_basin) && ~all(isnan(SW_basin(:, b)))
        sw_b = deseasonalize(SW_basin(:, b), target_dates);
    else
        sw_b = zeros(n_time, 1);
    end
    
    twsc_target = twsc_b;
    
    % Fill missing just in case
    X_nat = fillmissing([p_b, et_b, q_b], 'linear');
    X_anthro = fillmissing([p_b, et_b, q_b, gw_b, sw_b], 'linear');
    twsc_target = fillmissing(twsc_target, 'linear');
    
    % Prepare for LSTM: inputs must be Features x TimeSteps cell arrays
    % We train on the entire sequence as a single observation
    XTrainNat = {X_nat'};
    XTrainAnt = {X_anthro'};
    YTrain = {twsc_target'};
    
    %% --- Train Model 1 (M_nat) ---
    layers_nat = [
        sequenceInputLayer(size(X_nat, 2))
        lstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')
        lstmLayer(numHiddenUnits2, 'OutputMode', 'sequence')
        fullyConnectedLayer(1)
        regressionLayer];
        
    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'GradientThreshold', 1, ...
        'InitialLearnRate', 0.005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 50, ...
        'LearnRateDropFactor', 0.2, ...
        'Verbose', false, ...
        'Plots', 'none');
        
    % Ensemble for uncertainty
    num_ensemble = 3; % Small ensemble for computational feasibility
    y_preds_nat_ens = zeros(n_time, num_ensemble);
    for e = 1:num_ensemble
        net_nat = trainNetwork(XTrainNat, YTrain, layers_nat, options);
        y_pred_nat_seq = predict(net_nat, XTrainNat);
        y_preds_nat_ens(:, e) = y_pred_nat_seq{1}';
    end
    y_pred_nat = mean(y_preds_nat_ens, 2);
    y_stdev_nat = std(y_preds_nat_ens, 0, 2);
    
    res_nat = twsc_target - y_pred_nat;
    ss_tot = sum((twsc_target - mean(twsc_target)).^2);
    
    R2_nat(b)   = 1 - (sum(res_nat.^2) / ss_tot);
    RMSE_nat(b) = sqrt(mean(res_nat.^2));
    TWSC_pred_nat(:, b) = y_pred_nat;
    
    %% --- Train Model 2 (M_anthro) ---
    layers_ant = [
        sequenceInputLayer(size(X_anthro, 2))
        lstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')
        lstmLayer(numHiddenUnits2, 'OutputMode', 'sequence')
        fullyConnectedLayer(1)
        regressionLayer];
        
    y_preds_ant_ens = zeros(n_time, num_ensemble);
    for e = 1:num_ensemble
        net_ant_temp = trainNetwork(XTrainAnt, YTrain, layers_ant, options);
        y_pred_ant_seq = predict(net_ant_temp, XTrainAnt);
        y_preds_ant_ens(:, e) = y_pred_ant_seq{1}';
        if e == 1
            net_ant = net_ant_temp; % Save one network for feature importance/SHAP
        end
    end
    y_pred_ant = mean(y_preds_ant_ens, 2);
    y_stdev_ant = std(y_preds_ant_ens, 0, 2);
    
    res_ant = twsc_target - y_pred_ant;
    
    R2_anthro(b)   = 1 - (sum(res_ant.^2) / ss_tot);
    RMSE_anthro(b) = sqrt(mean(res_ant.^2));
    TWSC_pred_anthro(:, b) = y_pred_ant;
    
    %% --- Compute Uncertainty (95% CI) ---
    y_stdev_total = sqrt(y_stdev_nat.^2 + y_stdev_ant.^2);
    TWSC_pred_anthro_upper(:, b) = y_pred_ant + 1.96 * y_stdev_total;
    TWSC_pred_anthro_lower(:, b) = y_pred_ant - 1.96 * y_stdev_total;
    
    %% Feature Permutation Importance Scores
    feat_imp_b = zeros(1, 5);
    baseline_mse = mean((twsc_target - y_pred_ant).^2);
    for f = 1:5
        X_shuffled = X_anthro;
        X_shuffled(:, f) = X_shuffled(randperm(n_time), f);
        XTrainShuffled = {X_shuffled'};
        y_shuf_seq = predict(net_ant, XTrainShuffled);
        y_shuf = y_shuf_seq{1}';
        shuffled_mse = mean((twsc_target - y_shuf).^2);
        feat_imp_b(f) = shuffled_mse - baseline_mse; % Delta Error
    end
    feature_importance(b, :) = feat_imp_b;
    
    %% SHAP Computation (Event-level attribution)
    try
        % Custom predict wrapper for shapley
        customPredictFn = @(X_in) lstmPredictWrapper(net_ant, X_in);
        explainer = shapley(customPredictFn, X_anthro);
        shap_vals_table = explainer.ShapleyValues;
        shap_matrix = shap_vals_table{:, :};
        
        shap_b = nan(n_time, 5);
        shap_b(:, :) = shap_matrix;
        shap_values(:, b, :) = shap_b;
    catch
        % shapley might fail with custom sequence wrapper
    end
    
    %% Metrics
    Delta_R2(b) = R2_anthro(b) - R2_nat(b);
end

fprintf('\nLSTM Attribution modeling complete.\n');
fprintf('Average R^2 (Natural Model M_nat):     %.3f\n', mean(R2_nat, 'omitnan'));
fprintf('Average R^2 (Anthropogenic M_anthro):  %.3f\n', mean(R2_anthro, 'omitnan'));
fprintf('Average Delta R^2 (Anthropogenic Gain): %.3f\n', mean(Delta_R2, 'omitnan'));

%% Save Attribution Results to Disk
attr_file = fullfile(output_dir, 'attribution_results_lstm.mat');
fprintf('Saving LSTM attribution results to %s...\n', attr_file);
save(attr_file, 'R2_nat', 'R2_anthro', 'Delta_R2', 'RMSE_nat', 'RMSE_anthro', ...
    'feature_importance', 'shap_values', 'TWSC_obs', 'TWSC_pred_nat', 'TWSC_pred_anthro', ...
    'TWSC_pred_anthro_upper', 'TWSC_pred_anthro_lower', '-v7.3');
fprintf('=== STEP 4d Complete: Twin LSTM Model Attribution Completed & Saved ===\n\n');

%% Helper Function for Deseasonalization
function x_deseason = deseasonalize(x, dates)
    x_deseason = nan(size(x));
    valid_idx = ~isnan(x);
    baseline_idx = year(dates) >= 2004 & year(dates) <= 2009;
    
    if sum(valid_idx) < 24
        months = month(dates);
        for m = 1:12
            idx_m = (months == m);
            idx_baseline_m = idx_m & baseline_idx;
            if sum(~isnan(x(idx_baseline_m))) < 3
                monthly_mean = mean(x(idx_m), 'omitnan');
            else
                monthly_mean = mean(x(idx_baseline_m), 'omitnan');
            end
            x_deseason(idx_m) = x(idx_m) - monthly_mean;
        end
        return;
    end
    
    x_filled = x;
    if any(~valid_idx)
        x_filled = fillmissing(x, 'linear');
    end
    
    [LT, ST, R] = trenddecomp(x_filled, 'stl', 12);
    x_deseas_temp = LT + R;
    baseline_mean = mean(x_deseas_temp(baseline_idx), 'omitnan');
    x_deseason = x_deseas_temp - baseline_mean;
    x_deseason(~valid_idx) = NaN;
end

function Y = lstmPredictWrapper(net, X)
    % Wrapper for shapley to accept tabular data and convert to sequence cell
    % If X is N x Features, return N x 1
    X_cell = {X'};
    Y_seq = predict(net, X_cell);
    Y = Y_seq{1}';
end
