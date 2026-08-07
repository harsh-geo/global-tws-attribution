%% step04c_run_attribution_gridwise_mlr.m - Twin MLR Attribution (Grid-wise)
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% GOVERNING MASS BALANCE: dTWS/dt = TWSC = P - ET - Q - (GW_abs + SW_abs)
% =========================================================================
% PURPOSE:
%   1. Compute TWSC via centered finite differences per pixel.
%   2. Convert variables (P, ET, Q, GW_abs, SW_abs) to ANOMALIES.
%   3. Fit Twin Multiple Linear Regression (MLR) Models per pixel in parallel:
%      - Model 1 (Natural M_nat):     TWSC = b0 + b1*P + b2*ET + b3*Q
%      - Model 2 (Full M_anthro):     TWSC = b0 + b1*P + b2*ET + b3*Q + b4*GW + b5*SW
%   4. Compute Attribution Metrics for each pixel:
%      - Variance Explained Gain: Delta R^2 = R^2_anthro - R^2_nat
%      - Explicit Beta Coefficients
%
% OUTPUT:
%   - outputs/tables/attribution_results_gridwise.mat
% =========================================================================

fprintf('=== STEP 4c: Starting Twin MLR Attribution (Grid-wise) ===\n');

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

% Load reconstructed TWS
tws_file = fullfile(processed_dir, 'grace_gridwise_reconstructed.mat');
if exist(tws_file, 'file')
    fprintf('Loading reconstructed TWS from %s...\n', tws_file);
    load(tws_file);
else
    fprintf('Reconstructed TWS not found. Run step03b first.\n');
    return;
end

% Load original predictors
ts_file = fullfile(processed_dir, 'gridwise_time_series.mat');
if exist(ts_file, 'file')
    fprintf('Loading original predictors from %s...\n', ts_file);
    ts_data = load(ts_file);
    P_gridwise   = ts_data.P_gridwise;
    ET_gridwise  = ts_data.ET_gridwise;
    Q_gridwise   = ts_data.Q_gridwise;
    GW_gridwise  = ts_data.GW_gridwise;
    SW_gridwise  = ts_data.SW_gridwise;
    pixel_coords = ts_data.pixel_coords;
else
    fprintf('Predictors not found. Run step02b first.\n');
    return;
end

n_basins = length(TWS_reconstructed_gridwise);
[n_time, ~] = size(TWS_reconstructed_gridwise{1});

%% Baseline Indices for Anomalies
target_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';
baseline_idx = year(target_dates) >= 2004 & year(target_dates) <= 2008;

%% Initialize Result Cell Arrays
R2_nat_gridwise       = cell(n_basins, 1);
R2_anthro_gridwise    = cell(n_basins, 1);
Delta_R2_gridwise     = cell(n_basins, 1);

beta_nat_gridwise     = cell(n_basins, 1); % [4 x N_pixels]
beta_anthro_gridwise  = cell(n_basins, 1); % [6 x N_pixels]

%% Parallel Pool
num_cores = feature('numcores');
if isempty(gcp('nocreate'))
    parpool('local', num_cores);
end

fprintf('\nRunning Twin MLR Attribution Models in parallel across %d basins...\n', n_basins);

% Set up DataQueue for parfor progress
dq = parallel.pool.DataQueue;
if usejava('desktop')
    h_wb = waitbar(0, 'Running MLR Attribution...');
    afterEach(dq, @(~) update_waitbar(h_wb, n_basins));
else
    afterEach(dq, @(~) update_waitbar([], n_basins));
end

%% Parfor Loop over Basins
dt = 1.0;
parfor b = 1:n_basins
    tws_b = TWS_reconstructed_gridwise{b};
    p_b   = P_gridwise{b};
    et_b  = ET_gridwise{b};
    q_b   = Q_gridwise{b};
    gw_b  = GW_gridwise{b};
    sw_b  = SW_gridwise{b};
    
    n_pixels = size(tws_b, 2);
    
    % Initialize basin result vectors
    r2_n  = nan(1, n_pixels);
    r2_a  = nan(1, n_pixels);
    dr2   = nan(1, n_pixels);
    
    b_nat = nan(4, n_pixels); % [Intercept, P, ET, Q]
    b_ant = nan(6, n_pixels); % [Intercept, P, ET, Q, GW, SW]
    
    for p = 1:n_pixels
        tws_p = tws_b(:, p);
        p_p   = p_b(:, p);
        et_p  = et_b(:, p);
        q_p   = q_b(:, p);
        gw_p  = gw_b(:, p);
        sw_p  = sw_b(:, p);
        
        % Calculate TWSC via finite difference
        twsc_p = nan(n_time, 1);
        twsc_p(2:end-1) = (tws_p(3:end) - tws_p(1:end-2)) / (2 * dt);
        twsc_p(1)       = (tws_p(2) - tws_p(1)) / dt;
        twsc_p(end)     = (tws_p(end) - tws_p(end-1)) / dt;
        
        % Anomalies (2004-2008)
        p_p  = p_p - mean(p_p(baseline_idx), 'omitnan');
        et_p = et_p - mean(et_p(baseline_idx), 'omitnan');
        
        if ~isempty(q_p) && ~all(isnan(q_p))
            q_p = q_p - mean(q_p(baseline_idx), 'omitnan');
        else
            q_p = zeros(n_time, 1);
        end
        
        if ~isempty(gw_p) && ~all(isnan(gw_p))
            gw_p = gw_p - mean(gw_p(baseline_idx), 'omitnan');
        else
            gw_p = zeros(n_time, 1);
        end
        
        if ~isempty(sw_p) && ~all(isnan(sw_p))
            sw_p = sw_p - mean(sw_p(baseline_idx), 'omitnan');
        else
            sw_p = zeros(n_time, 1);
        end
        
        % Model 1: Natural (P, ET, Q)
        X_nat = [ones(n_time, 1), p_p, et_p, q_p];
        % Model 2: Anthropogenic (P, ET, Q, GW, SW)
        X_ant = [ones(n_time, 1), p_p, et_p, q_p, gw_p, sw_p];
        
        valid_mask = ~isnan(twsc_p) & ~any(isnan(X_ant), 2);
        if sum(valid_mask) < 30
            continue;
        end
        
        y_val     = twsc_p(valid_mask);
        X_nat_val = X_nat(valid_mask, :);
        X_ant_val = X_ant(valid_mask, :);
        
        % --- Fit M_nat ---
        beta_n = X_nat_val \ y_val;
        y_pred_n = X_nat_val * beta_n;
        ss_tot = sum((y_val - mean(y_val)).^2);
        ss_res_n = sum((y_val - y_pred_n).^2);
        r2_n_val = 1 - (ss_res_n / ss_tot);
        
        % --- Fit M_anthro ---
        beta_a = X_ant_val \ y_val;
        y_pred_a = X_ant_val * beta_a;
        ss_res_a = sum((y_val - y_pred_a).^2);
        r2_a_val = 1 - (ss_res_a / ss_tot);
        
        % Store
        b_nat(:, p) = beta_n;
        b_ant(:, p) = beta_a;
        r2_n(p) = r2_n_val;
        r2_a(p) = r2_a_val;
        dr2(p)  = r2_a_val - r2_n_val;
    end
    
    beta_nat_gridwise{b}    = b_nat;
    beta_anthro_gridwise{b} = b_ant;
    R2_nat_gridwise{b}      = r2_n;
    R2_anthro_gridwise{b}   = r2_a;
    Delta_R2_gridwise{b}    = dr2;
    
    % Update progress
    send(dq, b);
end

if exist('h_wb', 'var') && ~isempty(h_wb) && isvalid(h_wb)
    close(h_wb);
end

fprintf('\nAttribution complete.\n');

%% Save Results
attr_file = fullfile(output_dir, 'attribution_results_gridwise.mat');
fprintf('Saving grid-wise attribution results to %s...\n', attr_file);
save(attr_file, 'beta_nat_gridwise', 'beta_anthro_gridwise', ...
    'R2_nat_gridwise', 'R2_anthro_gridwise', 'Delta_R2_gridwise', 'pixel_coords', '-v7.3');
fprintf('=== STEP 4c Complete ===\n\n');

%% Helper Function for Parfor Progress
function update_waitbar(h_wb, n_total)
    persistent count;
    if isempty(count)
        count = 1;
    else
        count = count + 1;
    end
    
    if ~isempty(h_wb) && isvalid(h_wb)
        waitbar(count / n_total, h_wb, sprintf('Running Attribution basin %d / %d', count, n_total));
    elseif mod(count, 10) == 0 || count == 1 || count == n_total
        fprintf('  -> Attribution basin %d / %d...\n', count, n_total);
    end
    
    if count == n_total
        count = []; % Reset for next run
    end
end
