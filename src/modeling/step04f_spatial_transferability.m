%% step04f_spatial_transferability.m - Spatial Transferability Causal Test
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Calculate the mean total abstraction (GW_abs + SW_abs) for each basin.
%   2. Partition the 103 basins into "Pristine" (low abstraction) and 
%      "Irrigated" (high abstraction) groups.
%   3. Train a global Natural Random Forest (M_nat) exclusively on the 
%      Pristine basins.
%   4. Apply this model to predict TWSC in the Irrigated basins.
%   5. A systematic overprediction of TWSC (predicted > observed) in the 
%      Irrigated basins provides causal proof of the missing anthropogenic sink.
%
% OUTPUT:
%   - outputs/tables/transferability_results.mat
% =========================================================================

fprintf('=== STEP 4f: Starting Spatial Transferability Causal Test ===\n');

%% Directory Setup & Data Loading
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
processed_dir = fullfile(project_root, 'data', 'processed');
output_dir    = fullfile(project_root, 'outputs', 'tables');

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Load Basin Data
ts_file = fullfile(processed_dir, 'basin_time_series.mat');
ts_data = load(ts_file);
P_basin     = ts_data.P_basin;
ET_basin    = ts_data.ET_basin;
Q_basin     = ts_data.Q_basin;
GW_basin    = ts_data.GW_basin;
SW_basin    = ts_data.SW_basin;
grace_dates = ts_data.grace_dates;

% Load pre-computed attribution results (for TWSC_obs)
attr_file = fullfile(output_dir, 'attribution_results.mat');
attr_data = load(attr_file);
TWSC_obs  = attr_data.TWSC_obs;

[n_time, n_basins] = size(TWSC_obs);

% Define baseline for anomalies
target_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';
baseline_idx = year(target_dates) >= 2004 & year(target_dates) <= 2008;

%% 1. Partition Basins: Pristine vs. Irrigated
% Calculate mean total abstraction per basin
mean_abs = nan(1, n_basins);
for b = 1:n_basins
    if ~isempty(GW_basin) && ~all(isnan(GW_basin(:, b)))
        gw_mean = mean(GW_basin(:, b), 'omitnan');
    else
        gw_mean = 0;
    end
    if ~isempty(SW_basin) && ~all(isnan(SW_basin(:, b)))
        sw_mean = mean(SW_basin(:, b), 'omitnan');
    else
        sw_mean = 0;
    end
    mean_abs(b) = gw_mean + sw_mean;
end

% Define threshold for "Pristine" (e.g., bottom 50% of abstraction)
abs_threshold = median(mean_abs);
pristine_basins = find(mean_abs <= abs_threshold);
irrigated_basins = find(mean_abs > abs_threshold);

fprintf('Identified %d Pristine Basins and %d Irrigated Basins.\n', length(pristine_basins), length(irrigated_basins));

%% 2. Aggregate Training Data (Pristine Basins Only)
fprintf('Aggregating training data from Pristine basins...\n');
X_train = [];
Y_train = [];

for i = 1:length(pristine_basins)
    b = pristine_basins(i);
    
    twsc_target = deseasonalize(TWSC_obs(:, b), target_dates, baseline_idx);
    p_b  = deseasonalize(P_basin(:, b), target_dates, baseline_idx);
    et_b = deseasonalize(ET_basin(:, b), target_dates, baseline_idx);
    
    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q_b = deseasonalize(Q_basin(:, b), target_dates, baseline_idx);
    else
        q_b = zeros(n_time, 1);
    end
    
    X_nat = [p_b, et_b, q_b];
    valid_mask = ~isnan(twsc_target) & ~any(isnan(X_nat), 2);
    
    X_train = [X_train; X_nat(valid_mask, :)];
    Y_train = [Y_train; twsc_target(valid_mask)];
end

%% 3. Train Global Natural Model (M_nat_global)
fprintf('Training Global M_nat Random Forest on %d samples...\n', length(Y_train));
n_trees = 300; % Can be smaller for global model or standard 500
rf_nat_global = TreeBagger(n_trees, X_train, Y_train, ...
    'Method', 'regression', 'MinLeafSize', 5, 'NumPredictorsToSample', 1);

%% 4. Apply to Irrigated Basins and Calculate Causal Bias
fprintf('Applying M_nat_global to Irrigated basins...\n');

TWSC_pred_transfer = nan(n_time, n_basins);
bias_results = nan(1, n_basins);

for i = 1:length(irrigated_basins)
    b = irrigated_basins(i);
    
    twsc_target = deseasonalize(TWSC_obs(:, b), target_dates, baseline_idx);
    p_b  = deseasonalize(P_basin(:, b), target_dates, baseline_idx);
    et_b = deseasonalize(ET_basin(:, b), target_dates, baseline_idx);
    
    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q_b = deseasonalize(Q_basin(:, b), target_dates, baseline_idx);
    else
        q_b = zeros(n_time, 1);
    end
    
    X_nat = [p_b, et_b, q_b];
    valid_mask = ~any(isnan(X_nat), 2);
    
    if sum(valid_mask) > 0
        y_pred = predict(rf_nat_global, X_nat(valid_mask, :));
        TWSC_pred_transfer(valid_mask, b) = y_pred;
        
        % Calculate systematic bias: mean(TWSC_pred - TWSC_obs)
        % Positive bias means model predicts more water than observed (missing sink)
        valid_eval = valid_mask & ~isnan(twsc_target);
        if sum(valid_eval) > 0
            bias_results(b) = mean(y_pred(valid_eval(valid_mask)) - twsc_target(valid_eval), 'omitnan');
        end
    end
end

fprintf('Saving Transferability results...\n');
trans_file = fullfile(output_dir, 'transferability_results.mat');
save(trans_file, 'pristine_basins', 'irrigated_basins', 'TWSC_pred_transfer', 'bias_results', '-v7.3');
fprintf('=== STEP 4f Complete ===\n\n');

%% Helper Function
function x_deseason = deseasonalize(x, dates, baseline_idx)
    x_deseason = nan(size(x));
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
    end
end
