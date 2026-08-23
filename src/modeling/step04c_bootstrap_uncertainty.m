%% step04c_bootstrap_uncertainty.m - Block-Bootstrapping for Feature Importances & Delta R^2
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Moving / Contiguous Block-Bootstrapping (N = 1000 resamples per basin)
%      using 36-month (3-year) contiguous time blocks to preserve temporal
%      autocorrelation and hydrological memory.
%   2. Fit Twin Random Forest models across each bootstrap realization:
%      - Model 1 (Natural Baseline M_nat):     TWSC ~ f(P, ET, Q)
%      - Model 2 (Full Anthropogenic M_anthro): TWSC ~ f(P, ET, Q, GW_abs, SW_abs)
%   3. Compute Empirical Uncertainty & Confidence Intervals (95% CI):
%      - Delta R^2 distribution, mean, median, standard error (SE), 2.5th & 97.5th percentiles.
%      - Feature permutation importance distributions (P, ET, Q, GW_abs, SW_abs).
%      - Feature rank probabilities and stability metrics per basin.
%
% OUTPUTS:
%   - outputs/tables/bootstrap_uncertainty_results.mat
%   - outputs/tables/bootstrap_attribution_uncertainty.csv
% =========================================================================

fprintf('=== STEP 4c: Starting Block-Bootstrapping Uncertainty Quantification (N=1000) ===\n');

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

% 1. Load Reconstructed TWS
tws_file = fullfile(processed_dir, 'grace_reconstructed.mat');
if exist(tws_file, 'file')
    fprintf('Loading reconstructed TWS from %s...\n', tws_file);
    tws_data = load(tws_file);
    TWS = tws_data.TWS_reconstructed;
else
    error('Reconstructed TWS file not found: %s. Run step03_reconstruct_grace.m first.', tws_file);
end

% 2. Load Hydroclimate & Human Basin Time-Series
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

    if isfield(ts_data, 'T_basin')
        T_basin = ts_data.T_basin;
    else
        T_basin = zeros(size(P_basin));
    end
    if isfield(ts_data, 'ONI_index')
        ONI_index = ts_data.ONI_index;
    else
        ONI_index = zeros(size(P_basin, 1), 1);
    end
else
    error('Basin time series file not found: %s. Run step02_aggregate_basins.m first.', ts_file);
end

[n_time, n_basins] = size(TWS);
target_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';
fprintf('Loaded dataset: %d monthly timesteps | %d river basins.\n', n_time, n_basins);

%% Compute TWSC via Centered Finite Differences
dt = 1.0;
TWSC_obs = nan(n_time, n_basins);
for b = 1:n_basins
    tws_b = TWS(:, b);
    TWSC_obs(2:end-1, b) = (tws_b(3:end) - tws_b(1:end-2)) / (2 * dt);
    TWSC_obs(1, b)       = (tws_b(2) - tws_b(1)) / dt;
    TWSC_obs(end, b)     = (tws_b(end) - tws_b(end-1)) / dt;
end

%% Setup Parallel Processing
has_parallel = license('test', 'Distrib_Computing_Toolbox');
if has_parallel
    try
        if isempty(gcp('nocreate'))
            num_cores = feature('numcores');
            parpool('local', num_cores);
        end
    catch ME
        fprintf('Note: Running serially (Parallel pool warning: %s)\n', ME.message);
    end
end

%% Block Bootstrap Hyperparameters
N_boot      = 1000; % Restored to 1000 for robust CI estimation
block_len   = 36;   % 3-Year contiguous block size
n_trees     = 100;  % Number of trees per bootstrap forest (optimized for N=1000)
min_leaf    = 5;
n_features  = 5;    % P, ET, Q, GW_abs, SW_abs

% Number of blocks needed to construct a synthetic time-series of length n_time
num_blocks_needed = ceil(n_time / block_len);

% Preallocate Summary Result Arrays across all basins
delta_r2_mean   = nan(n_basins, 1);
delta_r2_median = nan(n_basins, 1);
delta_r2_se     = nan(n_basins, 1);
delta_r2_ci_low = nan(n_basins, 1);
delta_r2_ci_upp = nan(n_basins, 1);

feat_imp_mean   = nan(n_basins, n_features);
feat_imp_ci_low = nan(n_basins, n_features);
feat_imp_ci_upp = nan(n_basins, n_features);

top_driver_prob = nan(n_basins, n_features); % Probability of each feature being Rank 1

% Full bootstrap distributions: [N_boot x N_basins] and [N_boot x N_basins x n_features]
boot_delta_r2_dist  = nan(N_boot, n_basins);
boot_feat_imp_dist  = nan(N_boot, n_basins, n_features);

fprintf('\nStarting Block-Bootstrap iterations across %d basins (N_boot=%d, Block=%d months)...\n', ...
    n_basins, N_boot, block_len);

%% Parallel Basin Bootstrap Loop
% Setup DataQueue for parfor progress tracking
dq = parallel.pool.DataQueue;
afterEach(dq, @(b) eval('fprintf(''Basin %d complete.\n'', b); drawnow;'));
fprintf('Progress (Basin ID will print upon completion):\n');

parfor b = 1:n_basins
    twsc_b = deseasonalize_baseline(TWSC_obs(:, b), target_dates);
    p_b    = deseasonalize_baseline(P_basin(:, b), target_dates);
    et_b   = deseasonalize_baseline(ET_basin(:, b), target_dates);

    if ~isempty(Q_basin) && ~all(isnan(Q_basin(:, b)))
        q_b = deseasonalize_baseline(Q_basin(:, b), target_dates);
    else
        q_b = zeros(n_time, 1);
    end

    if ~isempty(GW_basin) && ~all(isnan(GW_basin(:, b)))
        gw_b = deseasonalize_baseline(GW_basin(:, b), target_dates);
    else
        gw_b = zeros(n_time, 1);
    end

    if ~isempty(SW_basin) && ~all(isnan(SW_basin(:, b)))
        sw_b = deseasonalize_baseline(SW_basin(:, b), target_dates);
    else
        sw_b = zeros(n_time, 1);
    end

    % Matrices
    X_nat_full = [p_b, et_b, q_b];
    X_ant_full = [p_b, et_b, q_b, gw_b, sw_b];
    y_full     = twsc_b;

    valid_mask = ~isnan(y_full) & ~any(isnan(X_ant_full), 2);
    if sum(valid_mask) < 36
        send(dq, b);
        continue;
    end

    % Candidate block start indices (Moving block bootstrap pool)
    max_start_idx = n_time - block_len + 1;

    % Local bootstrap storage for basin b
    b_delta_r2  = nan(N_boot, 1);
    b_feat_imp  = nan(N_boot, n_features);
    b_ranks     = nan(N_boot, n_features);

    % Bootstrap resample loop
    for iter = 1:N_boot
        % Draw random block start points with replacement
        block_starts = randi(max_start_idx, num_blocks_needed, 1);

        % Construct synthetic time-series indices
        sample_indices = [];
        for k = 1:num_blocks_needed
            sample_indices = [sample_indices; (block_starts(k) : block_starts(k) + block_len - 1)']; %#ok<AGROW>
        end
        sample_indices = sample_indices(1:n_time);

        % Extract resampled observations
        y_boot     = y_full(sample_indices);
        X_nat_boot = X_nat_full(sample_indices, :);
        X_ant_boot = X_ant_full(sample_indices, :);

        boot_valid = ~isnan(y_boot) & ~any(isnan(X_ant_boot), 2);
        if sum(boot_valid) < 30
            continue;
        end

        y_v   = y_boot(boot_valid);
        X_nat = X_nat_boot(boot_valid, :);
        X_ant = X_ant_boot(boot_valid, :);

        % 1. Train Model 1 (M_nat)
        rf_nat = TreeBagger(n_trees, X_nat, y_v, ...
            'Method', 'regression', ...
            'OOBPrediction', 'on', ...
            'MinLeafSize', min_leaf, ...
            'NumPredictorsToSample', 1);

        y_oob_nat = oobPredict(rf_nat);
        ss_tot    = sum((y_v - mean(y_v, 'omitnan')).^2, 'omitnan');
        r2_nat_i  = 1 - (sum((y_v - y_oob_nat).^2, 'omitnan') / ss_tot);

        % 2. Train Model 2 (M_anthro)
        rf_ant = TreeBagger(n_trees, X_ant, y_v, ...
            'Method', 'regression', ...
            'OOBPrediction', 'on', ...
            'OOBPredictorImportance', 'on', ...
            'MinLeafSize', min_leaf, ...
            'NumPredictorsToSample', 1);

        y_oob_ant = oobPredict(rf_ant);
        r2_ant_i  = 1 - (sum((y_v - y_oob_ant).^2, 'omitnan') / ss_tot);

        % Attribution Gain
        b_delta_r2(iter) = r2_ant_i - r2_nat_i;

        % Permutation Predictor Importance
        imp_scores = rf_ant.OOBPermutedPredictorDeltaError;
        b_feat_imp(iter, :) = imp_scores;

        % Rank features (1 = Highest importance)
        [~, sorted_idx] = sort(imp_scores, 'descend');
        ranks = zeros(1, n_features);
        for f = 1:n_features
            ranks(sorted_idx(f)) = f;
        end
        b_ranks(iter, :) = ranks;

        % Send progress dot per basin
    end

    % Emit progress dot to command window when basin completes
    send(dq, b);

    % Store distributions
    boot_delta_r2_dist(:, b)   = b_delta_r2;
    boot_feat_imp_dist(:, b, :) = b_feat_imp;

    % Calculate Empirical Statistics & 95% Confidence Intervals
    valid_res = ~isnan(b_delta_r2);
    min_required_boot = max(5, floor(N_boot * 0.5));
    if sum(valid_res) >= min_required_boot
        delta_r2_mean(b)   = mean(b_delta_r2(valid_res));
        delta_r2_median(b) = median(b_delta_r2(valid_res));
        delta_r2_se(b)     = std(b_delta_r2(valid_res));
        delta_r2_ci_low(b) = prctile(b_delta_r2(valid_res), 2.5);
        delta_r2_ci_upp(b) = prctile(b_delta_r2(valid_res), 97.5);

        for f = 1:n_features
            f_imp = b_feat_imp(valid_res, f);
            feat_imp_mean(b, f)   = mean(f_imp, 'omitnan');
            feat_imp_ci_low(b, f) = prctile(f_imp, 2.5);
            feat_imp_ci_upp(b, f) = prctile(f_imp, 97.5);

            % Probability that feature f is Rank 1
            top_driver_prob(b, f) = mean(b_ranks(valid_res, f) == 1);
        end
    end

end
fprintf('\n');

fprintf('\nBlock-Bootstrapping complete across all basins.\n');
fprintf('Average Bootstrap Delta R^2: %.4f (95%% CI Mean: [%.4f, %.4f])\n', ...
    mean(delta_r2_mean, 'omitnan'), mean(delta_r2_ci_low, 'omitnan'), mean(delta_r2_ci_upp, 'omitnan'));

%% Save MAT Results File
out_mat = fullfile(output_dir, 'bootstrap_uncertainty_results.mat');
fprintf('Saving bootstrap uncertainty results to %s...\n', out_mat);
save(out_mat, 'N_boot', 'block_len', 'boot_delta_r2_dist', 'boot_feat_imp_dist', ...
    'delta_r2_mean', 'delta_r2_median', 'delta_r2_se', 'delta_r2_ci_low', 'delta_r2_ci_upp', ...
    'feat_imp_mean', 'feat_imp_ci_low', 'feat_imp_ci_upp', 'top_driver_prob', '-v7.3');

%% Export Summary CSV Table
out_csv = fullfile(output_dir, 'bootstrap_attribution_uncertainty.csv');
fprintf('Exporting CSV summary table to %s...\n', out_csv);

feature_names = {'P', 'ET', 'Q', 'T', 'ONI', 'GW_abs', 'SW_abs'};

Basin_ID         = (1:n_basins)';
Delta_R2_Mean    = delta_r2_mean;
Delta_R2_Median  = delta_r2_median;
Delta_R2_SE      = delta_r2_se;
Delta_R2_CI_2_5  = delta_r2_ci_low;
Delta_R2_CI_97_5 = delta_r2_ci_upp;

Imp_P_Mean       = feat_imp_mean(:, 1);
Imp_P_CI_Low     = feat_imp_ci_low(:, 1);
Imp_P_CI_Upp     = feat_imp_ci_upp(:, 1);

Imp_ET_Mean      = feat_imp_mean(:, 2);
Imp_ET_CI_Low    = feat_imp_ci_low(:, 2);
Imp_ET_CI_Upp    = feat_imp_ci_upp(:, 2);

Imp_Q_Mean       = feat_imp_mean(:, 3);
Imp_Q_CI_Low     = feat_imp_ci_low(:, 3);
Imp_Q_CI_Upp     = feat_imp_ci_upp(:, 3);

Imp_T_Mean       = feat_imp_mean(:, 4);
Imp_T_CI_Low     = feat_imp_ci_low(:, 4);
Imp_T_CI_Upp     = feat_imp_ci_upp(:, 4);

Imp_ONI_Mean     = feat_imp_mean(:, 5);
Imp_ONI_CI_Low   = feat_imp_ci_low(:, 5);
Imp_ONI_CI_Upp   = feat_imp_ci_upp(:, 5);

Imp_GW_Mean      = feat_imp_mean(:, 6);
Imp_GW_CI_Low    = feat_imp_ci_low(:, 6);
Imp_GW_CI_Upp    = feat_imp_ci_upp(:, 6);

Imp_SW_Mean      = feat_imp_mean(:, 7);
Imp_SW_CI_Low    = feat_imp_ci_low(:, 7);
Imp_SW_CI_Upp    = feat_imp_ci_upp(:, 7);

Top_Driver_Prob_P  = top_driver_prob(:, 1);
Top_Driver_Prob_ET = top_driver_prob(:, 2);
Top_Driver_Prob_Q  = top_driver_prob(:, 3);
Top_Driver_Prob_T  = top_driver_prob(:, 4);
Top_Driver_Prob_ONI= top_driver_prob(:, 5);
Top_Driver_Prob_GW = top_driver_prob(:, 6);
Top_Driver_Prob_SW = top_driver_prob(:, 7);

T_summary = table(Basin_ID, Delta_R2_Mean, Delta_R2_Median, Delta_R2_SE, ...
    Delta_R2_CI_2_5, Delta_R2_CI_97_5, ...
    Imp_P_Mean, Imp_P_CI_Low, Imp_P_CI_Upp, ...
    Imp_ET_Mean, Imp_ET_CI_Low, Imp_ET_CI_Upp, ...
    Imp_Q_Mean, Imp_Q_CI_Low, Imp_Q_CI_Upp, ...
    Imp_T_Mean, Imp_T_CI_Low, Imp_T_CI_Upp, ...
    Imp_ONI_Mean, Imp_ONI_CI_Low, Imp_ONI_CI_Upp, ...
    Imp_GW_Mean, Imp_GW_CI_Low, Imp_GW_CI_Upp, ...
    Imp_SW_Mean, Imp_SW_CI_Low, Imp_SW_CI_Upp, ...
    Top_Driver_Prob_P, Top_Driver_Prob_ET, Top_Driver_Prob_Q, ...
    Top_Driver_Prob_T, Top_Driver_Prob_ONI, Top_Driver_Prob_GW, Top_Driver_Prob_SW);

writetable(T_summary, out_csv);
fprintf('=== STEP 4c Complete: Block-Bootstrapping Uncertainty Quantification Saved ===\n\n');

%% Helper Deseasonalization Function with Baseline
function x_deseason = deseasonalize_baseline(x, dates)
x_deseason = nan(size(x));
months = month(dates);
baseline_idx = year(dates) >= 2004 & year(dates) <= 2009;

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
end
