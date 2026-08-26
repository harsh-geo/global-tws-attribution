%% step04g_rf_shap_analysis.m - Formal Feature Attribution using SHAP
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Identify heavily depleted basins (e.g., Ganges-Brahmaputra, Tigris).
%   2. Re-train the Full Anthropogenic RF model (M_anthro) for these basins.
%   3. Compute Shapley (SHAP) values to isolate the directional impact
%      of groundwater and surface water abstraction on TWSC.
%
% OUTPUT:
%   - outputs/tables/shap_results.mat
% =========================================================================

fprintf('=== STEP 4g: Starting Formal Feature Attribution (SHAP) ===\n');

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
baseline_idx = year(target_dates) >= 2004 & year(target_dates) <= 2009;

%% Select Top Hotspot Basins for SHAP (To save compute)
% Basins 51 (Ganges), 39 (Tigris), 42 (Indus), 13 (Colorado), 40 (Hari Rud)
target_basins = [51, 39, 42, 13, 40]; 
n_targets = length(target_basins);

shap_results = cell(n_targets, 1);

%% Parfor Parallel Execution
num_cores = feature('numcores');
if isempty(gcp('nocreate'))
    parpool('local', num_cores);
end

parfor i = 1:n_targets
    b = target_basins(i);
    fprintf('Computing SHAP for Basin %d...\n', b);
    
    twsc_target = TWSC_obs(:, b);
    twsc_target = deseasonalize(twsc_target, target_dates, baseline_idx);
    
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
    
    % Model 2 (Anthropogenic): P, ET, Q, GW, SW
    X_anthro = [p_b, et_b, q_b, gw_b, sw_b];
    
    valid_mask = ~isnan(twsc_target) & ~any(isnan(X_anthro), 2);
    if sum(valid_mask) < 30
        continue;
    end
    
    y_val      = twsc_target(valid_mask);
    X_ant_val  = X_anthro(valid_mask, :);
    
    % Re-train M_anthro
    rf_anthro = TreeBagger(500, X_ant_val, y_val, ...
        'Method', 'regression', 'MinLeafSize', 5, 'NumPredictorsToSample', 1);
    
    predict_fcn = @(X) predict(rf_anthro, X);
    
    try
        explainer = shapley(predict_fcn, X_ant_val, 'QueryPoints', X_ant_val);
        shap_vals_table = explainer.Shapley;
        
        % Store results
        res = struct();
        res.basin_id = b;
        if istable(shap_vals_table)
            % shap_vals_table has features as rows and query points as columns.
            % The first column is 'Index' (strings), so we drop it.
            num_vals = double(shap_vals_table{:, 2:end}); 
            res.shap_values = num_vals'; % Transpose to get N x features
        else
            res.shap_values = shap_vals_table'; % Just in case it's numeric
        end
        res.valid_mask = valid_mask;
        res.X_ant_val = X_ant_val;
        shap_results{i} = res;
    catch ME
        fprintf('SHAP computation failed for basin %d: %s\n', b, ME.message);
    end
end

fprintf('Saving SHAP results...\n');
shap_file = fullfile(output_dir, 'shap_results.mat');
save(shap_file, 'shap_results', 'target_basins', '-v7.3');
fprintf('=== STEP 4g Complete ===\n\n');

%% Helper Function
function x_deseason = deseasonalize(x, dates, baseline_idx)
    x_deseason = nan(size(x));
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
