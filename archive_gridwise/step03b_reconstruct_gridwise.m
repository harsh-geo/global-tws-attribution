%% step03b_reconstruct_gridwise.m - ML Reconstruction of GRACE Observational Gaps
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Load grid-wise time-series from 'data/processed/gridwise_time_series.mat'.
%   2. Identify missing observational months in GRACE TWS anomalies.
%   3. Fit a Machine Learning (Random Forest) model per pixel using hydroclimate
%      predictors (P, ET, Q) during continuous overlapping periods.
%   4. Predict TWS anomalies for all missing gap months across 2002-2021.
%   5. HPC Parallel Execution (GEMINI.md Directive): Enforce 'parfor'
%      parallelization across all 103 river basins.
%
% OUTPUT:
%   - data/processed/grace_gridwise_reconstructed.mat containing:
%     TWS_reconstructed_gridwise (Cell array 103x1 of N_time x N_pixels), grace_dates
% =========================================================================

fprintf('=== STEP 3b: Starting ML (RF) GRACE Gap-Filling (Grid-wise) ===\n');

%% Directory & Data Loading
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

processed_dir = fullfile(project_root, 'data', 'processed');

if ~exist('TWS_gridwise', 'var') || isempty(TWS_gridwise)
    input_mat = fullfile(processed_dir, 'gridwise_time_series.mat');
    if exist(input_mat, 'file')
        fprintf('Loading grid-wise time-series from %s...\n', input_mat);
        load(input_mat);
    else
        fprintf('Grid-wise time-series not in memory. Run step02b first.\n');
        return;
    end
end

n_basins = length(TWS_gridwise);
[n_time, ~] = size(TWS_gridwise{1});

%% Initialize Output Structures
TWS_reconstructed_gridwise = TWS_gridwise;

%% Setup MATLAB Parallel Pool
num_cores = feature('numcores');
if isempty(gcp('nocreate'))
    fprintf('Initializing Parallel Pool with %d workers...\n', num_cores);
    try
        parpool('local', num_cores);
    catch
        parpool('local');
    end
end

% Random Forest Hyperparameters
n_trees       = 100;  % Reduced to 100 for grid-wise computational feasibility
min_leaf_size = 5;

fprintf('\nRunning ML (Random Forest) gap-filling in parallel across %d basins...\n', n_basins);

% Set up DataQueue for parfor progress
dq = parallel.pool.DataQueue;
if usejava('desktop')
    h_wb = waitbar(0, 'Gap-filling grid-wise data (RF)...');
    afterEach(dq, @(~) update_waitbar(h_wb, n_basins));
else
    afterEach(dq, @(~) update_waitbar([], n_basins));
end

%% Parfor Parallel Execution Loop Across 103 River Basins
parfor b = 1:n_basins
    tws_b = TWS_gridwise{b};
    p_b   = P_gridwise{b};
    et_b  = ET_gridwise{b};
    q_b   = Q_gridwise{b};
    
    n_pixels = size(tws_b, 2);
    
    % Temporary reconstructed matrix for this basin
    tws_recon_b = tws_b;
    
    % Iterate through pixels in this basin
    for p = 1:n_pixels
        tws_p = tws_b(:, p);
        p_p   = p_b(:, p);
        et_p  = et_b(:, p);
        q_p   = q_b(:, p);
        
        if all(isnan(p_p)) || all(isnan(et_p))
            continue;
        end
        
        % Predictors: P, ET, Q, and (P - ET - Q) as residual
        if ~isempty(q_p) && ~all(isnan(q_p))
            p_minus_et_q = p_p - et_p - q_p;
            X_all = [p_p, et_p, q_p, p_minus_et_q];
        else
            p_minus_et = p_p - et_p;
            X_all = [p_p, et_p, p_minus_et];
        end
        
        % Identify valid training months
        train_mask = ~isnan(tws_p) & ~any(isnan(X_all), 2);
        
        % Identify gap months to predict
        predict_mask = isnan(tws_p) & ~any(isnan(X_all), 2);
        
        if sum(train_mask) < 20
            % Insufficient baseline observations for training
            continue;
        end
        
        X_train = X_all(train_mask, :);
        y_train = tws_p(train_mask);
        
        % Fit Random Forest model
        rf_model = TreeBagger(n_trees, X_train, y_train, ...
            'Method', 'regression', ...
            'MinLeafSize', min_leaf_size, ...
            'OOBPrediction', 'off', ...
            'OOBPredictorImportance', 'off');
        
        % Predict TWS for missing gap months
        if any(predict_mask)
            X_predict = X_all(predict_mask, :);
            y_pred = predict(rf_model, X_predict);
            
            % Insert predictions
            tws_recon_b(predict_mask, p) = y_pred;
        end
    end
    
    TWS_reconstructed_gridwise{b} = tws_recon_b;
    
    % Update progress
    send(dq, b);
end

if exist('h_wb', 'var') && ~isempty(h_wb) && isvalid(h_wb)
    close(h_wb);
end

fprintf('\nML reconstruction completed for all %d basins.\n', n_basins);

%% Save Reconstructed Continuous TWS Time-Series to Disk
tws_file = fullfile(processed_dir, 'grace_gridwise_reconstructed.mat');
fprintf('Saving reconstructed TWS to %s...\n', tws_file);
save(tws_file, 'TWS_reconstructed_gridwise', 'grace_dates', '-v7.3');
fprintf('=== STEP 3b Complete ===\n\n');

%% Helper Function for Parfor Progress
function update_waitbar(h_wb, n_total)
    persistent count;
    if isempty(count)
        count = 1;
    else
        count = count + 1;
    end
    
    if ~isempty(h_wb) && isvalid(h_wb)
        waitbar(count / n_total, h_wb, sprintf('Gap-filling basin %d / %d', count, n_total));
    elseif mod(count, 10) == 0 || count == 1 || count == n_total
        fprintf('  -> Gap-filling basin %d / %d...\n', count, n_total);
    end
    
    if count == n_total
        count = []; % Reset for next run
    end
end

