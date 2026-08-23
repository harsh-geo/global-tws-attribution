%% step03_reconstruct_grace.m - Random Forest Reconstruction of GRACE Observational Gaps
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Load aggregated basin time-series from 'data/processed/basin_time_series.mat'.
%   2. Identify missing observational months in GRACE TWS anomalies (including
%      the 2017-2018 observational gap between GRACE and GRACE-FO).
%   3. Train an ensemble Random Forest regressor per basin using hydroclimate
%      predictors (Precipitation P, Evapotranspiration ET, Runoff Q) during
%      continuous overlapping observational baseline periods (2002-2017).
%   4. Predict TWS anomalies for all missing gap months across 2002-2021.
%   5. HPC Parallel Execution (GEMINI.md Directive): Enforce 'parfor'
%      parallelization across all 103 river basins.
%
% OUTPUT:
%   - data/processed/grace_reconstructed.mat containing:
%     TWS_reconstructed (N_time x 103 continuous matrix), grace_dates, oob_rmse
% =========================================================================

fprintf('=== STEP 3: Starting Random Forest GRACE Gap-Filling & Reconstruction ===\n');

%% Directory & Data Loading
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

processed_dir = fullfile(project_root, 'data', 'processed');

% Check if basin time-series variables are in memory
if ~exist('P_basin', 'var') || isempty(P_basin) || ~exist('TWS_basin', 'var') || isempty(TWS_basin)
    input_mat = fullfile(processed_dir, 'basin_time_series.mat');
    if exist(input_mat, 'file')
        fprintf('Loading basin time-series from %s...\n', input_mat);
        data = load(input_mat);
        P_basin     = data.P_basin;
        ET_basin    = data.ET_basin;
        Q_basin     = data.Q_basin;
        if isfield(data, 'GW_basin'), GW_basin = data.GW_basin; end
        if isfield(data, 'SW_basin'), SW_basin = data.SW_basin; end
        if isfield(data, 'T_basin'), T_basin = data.T_basin; end
        if isfield(data, 'ONI_index'), ONI_index = data.ONI_index; end
        TWS_basin   = data.TWS_basin;
        grace_dates = data.grace_dates;
    else
        fprintf('Basin time-series not in memory. Running step02_aggregate_basins.m...\n');
        run(fullfile(project_root, 'src', 'preprocessing', 'step02_aggregate_basins.m'));
    end
end

[n_time, n_basins] = size(P_basin);
fprintf('Time series length: %d months | Total Basins: %d\n', n_time, n_basins);

%% Initialize Output Structures
TWS_reconstructed = TWS_basin; % Will be populated with predicted values at NaN gaps
oob_rmse          = nan(1, n_basins);
oob_r2            = nan(1, n_basins);

%% Setup MATLAB Parallel Pool (HPC SLURM Integration)
num_cores = feature('numcores');
if isempty(gcp('nocreate'))
    fprintf('Initializing Parallel Pool with %d workers...\n', num_cores);
    try
        parpool('local', num_cores);
    catch
        parpool('local');
    end
end

%% Random Forest Hyperparameters
n_trees       = 200;  % Number of decision trees in Random Forest
min_leaf_size = 5;    % Minimum leaf node size for regularization

fprintf('\nRunning Random Forest gap-filling in parallel across %d basins...\n', n_basins);

%% Parfor Parallel Execution Loop Across 103 River Basins
parfor b = 1:n_basins
    tws_b = TWS_basin(:, b);
    p_b   = P_basin(:, b);
    et_b  = ET_basin(:, b);
    q_b   = Q_basin(:, b);
    
    if exist('T_basin', 'var') && size(T_basin, 2) >= b
        t_b = T_basin(:, b);
    else
        t_b = zeros(n_time, 1);
    end
    
    if exist('ONI_index', 'var')
        oni_b = ONI_index;
    else
        oni_b = zeros(n_time, 1);
    end

    % Check if predictor data exists
    if all(isnan(p_b)) || all(isnan(et_b))
        warning('Basin %d has missing predictor data. Skipping reconstruction.', b);
        continue;
    end

    % Compute hydroclimate water balance residual as additional predictor: P - ET - Q
    if ~isempty(q_b) && ~all(isnan(q_b))
        p_minus_et_q = p_b - et_b - q_b;
        X_all = [p_b, et_b, q_b, p_minus_et_q, t_b, oni_b];
    else
        p_minus_et = p_b - et_b;
        X_all = [p_b, et_b, p_minus_et, t_b, oni_b];
    end

    % Identify valid training months: GRACE TWS is observed AND predictors are valid
    train_mask = ~isnan(tws_b) & ~any(isnan(X_all), 2);

    % Identify gap months to predict: GRACE TWS is NaN BUT predictors are valid
    predict_mask = isnan(tws_b) & ~any(isnan(X_all), 2);

    if sum(train_mask) < 20
        % Insufficient baseline observations for training
        continue;
    end

    X_train = X_all(train_mask, :);
    y_train = tws_b(train_mask);

    % Train Random Forest Ensemble Regressor using TreeBagger
    rf_model = TreeBagger(n_trees, X_train, y_train, ...
        'Method', 'regression', ...
        'OOBPrediction', 'on', ...
        'MinLeafSize', min_leaf_size, ...
        'OOBPredictorImportance', 'off');

    % Evaluate Out-of-Bag (OOB) Performance on baseline training data
    y_oob = oobPredict(rf_model);
    res   = y_train - y_oob;
    oob_rmse(b) = sqrt(mean(res.^2, 'omitnan'));

    SS_tot = sum((y_train - mean(y_train, 'omitnan')).^2);
    SS_res = sum(res.^2, 'omitnan');
    oob_r2(b) = 1 - (SS_res / SS_tot);

    % Predict TWS for missing gap months
    if any(predict_mask)
        X_predict = X_all(predict_mask, :);
        y_pred = predict(rf_model, X_predict);

        % Insert reconstructed predictions into final TWS matrix
        tws_reconstructed_b = tws_b;
        tws_reconstructed_b(predict_mask) = y_pred;
        TWS_reconstructed(:, b) = tws_reconstructed_b;
    end
end

fprintf('\nRandom Forest reconstruction completed for all %d basins.\n', n_basins);
fprintf('Mean Out-of-Bag (OOB) RMSE across basins: %.3f cm\n', mean(oob_rmse, 'omitnan'));
fprintf('Mean Out-of-Bag (OOB) R^2 across basins:  %.3f\n', mean(oob_r2, 'omitnan'));

%% Save Reconstructed Continuous TWS Time-Series to Disk
tws_file = fullfile(processed_dir, 'grace_reconstructed.mat');
fprintf('Saving reconstructed TWS to %s...\n', tws_file);
save(tws_file, 'TWS_reconstructed', 'grace_dates', 'oob_rmse', 'oob_r2', '-v7.3');
fprintf('=== STEP 3 Complete: Continuous TWS Dataset Reconstructed & Saved ===\n\n');
