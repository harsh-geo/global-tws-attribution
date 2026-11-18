%% recompute_rf_tws_metrics.m
% =========================================================================
% Recompute TWS integration and R^2 / Delta_R2 from trained TWSC predictions
% Fixes the NaN-bias issue in step04_run_attribution that zeroed out R2 metrics.
% =========================================================================

cd('/home/harsh/SILIKA/Thesis/data2');
project_root = pwd;

table_dir = fullfile(project_root, 'outputs', 'tables');
proc_dir  = fullfile(project_root, 'data', 'processed');

fprintf('Loading data...\n');
attr_mat = fullfile(table_dir, 'attribution_results.mat');
attr_data = load(attr_mat);

tws_mat  = fullfile(proc_dir, 'grace_reconstructed.mat');
tws_data = load(tws_mat);
TWS = tws_data.TWS_reconstructed;

TWSC_obs = attr_data.TWSC_obs;
TWSC_nat = attr_data.TWSC_pred_nat_all;
TWSC_ant = attr_data.TWSC_pred_anthro_all;

[n_time, n_basins] = size(TWS);

R2_TWS_nat          = nan(1, n_basins);
R2_TWS_anthro       = nan(1, n_basins);
Delta_R2            = nan(1, n_basins);
TWS_pred_nat_all    = nan(n_time, n_basins);
TWS_pred_anthro_all = nan(n_time, n_basins);

for b = 1:n_basins
    idx_start = find(~isnan(TWS(:, b)), 1);
    if isempty(idx_start)
        continue;
    end
    
    twsc_nat_raw = TWSC_nat(:, b);
    twsc_ant_raw = TWSC_ant(:, b);
    
    % Ensure valid mask explicitly excludes NaNs from TWSC_obs (e.g. index 1)
    valid = ~isnan(TWS(:, b)) & ~isnan(twsc_nat_raw) & ~isnan(TWSC_obs(:, b));
    if sum(valid) < 30
        continue;
    end
    
    bias_nat = mean(twsc_nat_raw(valid)) - mean(TWSC_obs(valid, b));
    bias_ant = mean(twsc_ant_raw(valid)) - mean(TWSC_obs(valid, b));
    
    twsc_nat_corr = twsc_nat_raw - bias_nat;
    twsc_ant_corr = twsc_ant_raw - bias_ant;
    
    twsc_nat_corr(isnan(twsc_nat_corr)) = 0;
    twsc_ant_corr(isnan(twsc_ant_corr)) = 0;
    
    tws_pred_nat = nan(n_time, 1);
    tws_pred_ant = nan(n_time, 1);
    tws_pred_nat(idx_start:end) = TWS(idx_start, b) + cumsum(twsc_nat_corr(idx_start:end));
    tws_pred_ant(idx_start:end) = TWS(idx_start, b) + cumsum(twsc_ant_corr(idx_start:end));
    
    TWS_pred_nat_all(:, b)    = tws_pred_nat;
    TWS_pred_anthro_all(:, b) = tws_pred_ant;
    
    valid_tws = ~isnan(TWS(:, b)) & ~isnan(tws_pred_nat) & ~isnan(tws_pred_ant);
    if sum(valid_tws) > 30
        r_nat = corr(TWS(valid_tws, b), tws_pred_nat(valid_tws), 'rows', 'complete');
        r_ant = corr(TWS(valid_tws, b), tws_pred_ant(valid_tws), 'rows', 'complete');
        R2_TWS_nat(b)    = r_nat^2;
        R2_TWS_anthro(b) = r_ant^2;
        Delta_R2(b)      = r_ant^2 - r_nat^2;
    end
end

fprintf('\n=== Recomputed Summary ===\n');
fprintf('Basins evaluated: %d\n', sum(~isnan(Delta_R2)));
fprintf('R2_TWS_nat:    mean = %.4f (range: %.4f to %.4f)\n', mean(R2_TWS_nat, 'omitnan'), min(R2_TWS_nat), max(R2_TWS_nat));
fprintf('R2_TWS_anthro: mean = %.4f (range: %.4f to %.4f)\n', mean(R2_TWS_anthro, 'omitnan'), min(R2_TWS_anthro), max(R2_TWS_anthro));
fprintf('Delta_R2:      mean = %.4f (range: %.4f to %.4f)\n', mean(Delta_R2, 'omitnan'), min(Delta_R2), max(Delta_R2));
fprintf('Basins with Delta_R2 > 0: %d / %d (%.1f%%)\n', sum(Delta_R2 > 0), sum(~isnan(Delta_R2)), 100*sum(Delta_R2 > 0)/sum(~isnan(Delta_R2)));

% 1. Save updated attribution_results.mat
fprintf('\nUpdating %s...\n', attr_mat);
save(attr_mat, 'R2_TWS_nat', 'R2_TWS_anthro', 'Delta_R2', ...
    'TWS_pred_nat_all', 'TWS_pred_anthro_all', '-append');
fprintf('✓ attribution_results.mat updated successfully.\n');

% 2. Save updated validation_and_trends.mat
trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
if exist(trend_mat, 'file')
    fprintf('Updating %s...\n', trend_mat);
    R2_nat    = R2_TWS_nat;
    R2_anthro = R2_TWS_anthro;
    save(trend_mat, 'R2_nat', 'R2_anthro', 'Delta_R2', '-append');
    fprintf('✓ validation_and_trends.mat updated successfully.\n');
end

fprintf('\nAll RF metrics successfully recomputed and saved.\n');
