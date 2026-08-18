%% plot_basin_51_analysis.m
% =========================================================================
% PURPOSE: Create three specific figures for Basin 51 (Ganga Basin):
%   1. Scatter plot of OOB predicted vs observed TWS anomalies (w/ R2, RMSE)
%   2. Time series of observed TWS and reconstructed TWS, highlighting gap.
%   3. Histogram/Boxplot of OOB R2 values across all 103 basins.
% =========================================================================

% 1. Set paths and load data
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

table_dir = fullfile(project_root, 'outputs', 'tables');
out_dir = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
if ~exist(trend_mat, 'file')
    error('File validation_and_trends.mat not found.');
end

fprintf('Loading data...\n');
load(trend_mat, 'TWS_basin', 'P_basin', 'ET_basin', 'Q_basin', ...
    'TWS_reconstructed', 'grace_dates', 'oob_r2', 'oob_rmse');

basin_idx = 51;

%% Figure 1: Train RF for Basin 51 to get OOB predictions and create scatter plot
tws_b = TWS_basin(:, basin_idx);
p_b   = P_basin(:, basin_idx);
et_b  = ET_basin(:, basin_idx);
q_b   = Q_basin(:, basin_idx);

if ~isempty(q_b) && ~all(isnan(q_b))
    p_minus_et_q = p_b - et_b - q_b;
    X_all = [p_b, et_b, q_b, p_minus_et_q];
else
    p_minus_et = p_b - et_b;
    X_all = [p_b, et_b, p_minus_et];
end

train_mask = ~isnan(tws_b) & ~any(isnan(X_all), 2);
X_train = X_all(train_mask, :);
y_train = tws_b(train_mask);

% Train RF (same hyperparameters as step03)
n_trees = 200;
min_leaf_size = 5;
rng(42); % For reproducibility
rf_model = TreeBagger(n_trees, X_train, y_train, ...
    'Method', 'regression', ...
    'OOBPrediction', 'on', ...
    'MinLeafSize', min_leaf_size, ...
    'OOBPredictorImportance', 'off');

y_oob = oobPredict(rf_model);

% Plot 1
figure('Name', 'Basin 51 OOB Scatter', 'Color', 'w', 'Position', [100, 100, 600, 500]);
scatter(y_train, y_oob, 30, 'MarkerEdgeColor', [0, 0.4470, 0.7410], 'MarkerFaceColor', [0.3010, 0.7450, 0.9330], 'MarkerFaceAlpha', 0.6);
hold on;
% 1:1 Reference line
min_val = min([y_train; y_oob]);
max_val = max([y_train; y_oob]);
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);

xlabel('Observed TWS Anomaly (cm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('OOB Predicted TWS Anomaly (cm)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Basin %d (Ganga): Out-of-Bag Prediction vs Observed', basin_idx), 'FontSize', 14);
grid on;
box on;

% Annotate R2 and RMSE
r2_val = oob_r2(basin_idx);
rmse_val = oob_rmse(basin_idx);
annot_str = sprintf('R^2 = %.2f\nRMSE = %.2f cm', r2_val, rmse_val);
text(0.05, 0.95, annot_str, 'Units', 'normalized', 'FontSize', 12, ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'w', 'EdgeColor', 'k');

saveas(gcf, fullfile(out_dir, 'basin_51_scatter.png'));
fprintf('Saved Basin 51 Scatter Plot.\n');

%% Figure 2: Time Series Plot
figure('Name', 'Basin 51 Time Series', 'Color', 'w', 'Position', [150, 150, 900, 400]);
hold on;

% Define 2017-2018 gap (July 2017 to May 2018)
gap_start = datetime(2017, 7, 1);
gap_end   = datetime(2018, 5, 1);

% Shaded region for gap
ylim_vals = [min(TWS_reconstructed(:, basin_idx)) - 2, max(TWS_reconstructed(:, basin_idx)) + 2];
patch([gap_start gap_end gap_end gap_start], [ylim_vals(1) ylim_vals(1) ylim_vals(2) ylim_vals(2)], ...
    [0.9 0.9 0.9], 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Plot reconstructed line
h1 = plot(grace_dates, TWS_reconstructed(:, basin_idx), '-', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5, 'DisplayName', 'Reconstructed TWS');

% Plot observed markers
h2 = plot(grace_dates, TWS_basin(:, basin_idx), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0, 0.4470, 0.7410], 'MarkerSize', 5, 'DisplayName', 'Observed TWS');

ylim(ylim_vals);
xlim([min(grace_dates), max(grace_dates)]);
xlabel('Time', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('TWS Anomaly (cm)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Basin %d (Ganga): TWS Anomaly Time Series', basin_idx), 'FontSize', 14);
legend([h2, h1], 'Location', 'best');
grid on;
box on;
set(gca, 'Layer', 'top'); % Bring axes lines on top of the patch

saveas(gcf, fullfile(out_dir, 'basin_51_timeseries.png'));
fprintf('Saved Basin 51 Time Series Plot.\n');

%% Figure 3: Histogram of OOB R2 across all basins
figure('Name', 'Global OOB R2 Distribution', 'Color', 'w', 'Position', [200, 200, 600, 450]);
h = histogram(oob_r2, 15, 'FaceColor', [0.4660, 0.6740, 0.1880], 'EdgeColor', 'k', 'FaceAlpha', 0.8);
xlabel('Out-of-Bag R^2', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Number of Basins', 'FontSize', 12, 'FontWeight', 'bold');
title('Distribution of RF Out-of-Bag R^2 (All 103 Basins)', 'FontSize', 14);
grid on;
box on;

% Add a vertical line for mean R2
mean_r2 = mean(oob_r2, 'omitnan');
hold on;
plot([mean_r2, mean_r2], ylim, 'r--', 'LineWidth', 2);
text(mean_r2 + 0.02, max(ylim)*0.9, sprintf('Mean R^2 = %.2f', mean_r2), 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');

saveas(gcf, fullfile(out_dir, 'global_oob_r2_histogram.png'));
fprintf('Saved Global OOB R2 Histogram.\n');

fprintf('All figures generated successfully.\n');
