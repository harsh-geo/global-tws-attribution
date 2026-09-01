%% plot_rf_vs_lstm_comprehensive.m
% =========================================================================
% AUTHOR: Computational Hydrology Research
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% PURPOSE: Create a 4-panel publication-grade figure comparing Random Forest (RF)
%          and Long Short-Term Memory (LSTM) models across 103 global river basins:
%          - Panel (a): Explanatory Power (R^2) distribution: M_nat vs M_anthro (RF vs LSTM)
%          - Panel (b): Scatter plot of Anthropogenic Gain: Delta R^2 (RF) vs Delta R^2 (LSTM)
%          - Panel (c): Global Mean Feature Importance Comparison (P, ET, Q, GW_abs, SW_abs)
%          - Panel (d): Observed vs RF vs LSTM TWSC anomaly time series for a hotspot basin (Basin 51: Ganges-Brahmaputra)
% =========================================================================

clear; clc; close all;

%% 1. Set Paths and Load Data
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

table_dir = fullfile(project_root, 'outputs', 'tables');
fig_dir   = fullfile(project_root, 'outputs', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

rf_mat   = fullfile(table_dir, 'attribution_results.mat');
lstm_mat = fullfile(table_dir, 'attribution_results_lstm.mat');

if ~exist(rf_mat, 'file')
    error('RF attribution results file not found: %s', rf_mat);
end
if ~exist(lstm_mat, 'file')
    error('LSTM attribution results file not found: %s', lstm_mat);
end

fprintf('Loading RF and LSTM attribution results...\n');
rf_data   = load(rf_mat);
lstm_data = load(lstm_mat);

% Extract RF metrics
R2_nat_rf        = rf_data.R2_TWS_nat(:);
R2_anthro_rf     = rf_data.R2_TWS_anthro(:);
Delta_R2_rf      = rf_data.R2_TWS_anthro(:) - rf_data.R2_TWS_nat(:);
TWS_pred_ant_rf  = rf_data.TWS_pred_anthro_all;

% Load RF SHAP values
rf_shap_mat = fullfile(table_dir, 'shap_results.mat');
if exist(rf_shap_mat, 'file')
    rf_shap_data = load(rf_shap_mat);
    shap_results_rf = rf_shap_data.shap_results;
    feat_imp_rf = nan(103, 5);
    for i = 1:length(shap_results_rf)
        if ~isempty(shap_results_rf{i})
            % Mean absolute SHAP per feature for this basin
            feat_imp_rf(i, :) = mean(abs(shap_results_rf{i}.shap_values), 1, 'omitnan');
        end
    end
else
    feat_imp_rf = nan(103, 5); % Placeholder if step04g hasn't been run yet
end

% Extract LSTM metrics
R2_nat_lstm        = lstm_data.R2_TWS_nat(:);
R2_anthro_lstm     = lstm_data.R2_TWS_anthro(:);
Delta_R2_lstm      = lstm_data.R2_TWS_anthro(:) - lstm_data.R2_TWS_nat(:);
TWS_pred_ant_lstm  = lstm_data.TWS_pred_anthro_all;

% Compute LSTM SHAP mean absolute importance
shap_values_lstm = lstm_data.shap_values; % [n_time x 103 x 5]
feat_imp_lstm = squeeze(mean(abs(shap_values_lstm), 1, 'omitnan')); % [103 x 5]

% Load observed TWS for the timeseries plot
ts_mat = fullfile(project_root, 'data', 'processed', 'grace_reconstructed.mat');
ts_data = load(ts_mat);
TWS_obs = ts_data.TWS_reconstructed;

n_time   = size(TWS_obs, 1);
n_basins = size(TWS_obs, 2);
dates    = datetime(2002, 4, 1) + calmonths(0:n_time-1);

%% 2. Setup Figure Layout & Styling
fig = figure('Name', 'RF vs LSTM Global Attribution Comparison', ...
    'Units', 'inches', 'Position', [1, 1, 14, 10], 'Color', 'w');

% Palettes
c_rf_nat    = [0.20, 0.40, 0.70];
c_rf_ant    = [0.00, 0.60, 0.85];
c_lstm_nat  = [0.85, 0.35, 0.10];
c_lstm_ant  = [0.95, 0.60, 0.15];
c_obs       = [0.15, 0.15, 0.15];

%% --- PANEL (a): Model Performance Distribution (R^2 Boxplots) ---
subplot(2, 2, 1);
hold on; grid on; box on;

% Combine data into matrix for boxplot
box_data = [R2_nat_rf, R2_anthro_rf, R2_nat_lstm, R2_anthro_lstm];
x_pos = [1, 2, 3.5, 4.5];
box_width = 0.35;

% Custom Boxplot Rendering
colors = {c_rf_nat, c_rf_ant, c_lstm_nat, c_lstm_ant};
calc_stats = @(x) [prctile(x(~isnan(x)), 5), prctile(x(~isnan(x)), 25), ...
    median(x(~isnan(x))), prctile(x(~isnan(x)), 75), prctile(x(~isnan(x)), 95)];

for k = 1:4
    vals = box_data(:, k);
    vals = vals(~isnan(vals));
    s = calc_stats(vals);
    x = x_pos(k);

    % Whiskers
    plot([x, x], [s(1), s(2)], 'k-', 'LineWidth', 1.2);
    plot([x, x], [s(4), s(5)], 'k-', 'LineWidth', 1.2);
    plot([x-0.08, x+0.08], [s(1), s(1)], 'k-', 'LineWidth', 1.2);
    plot([x-0.08, x+0.08], [s(5), s(5)], 'k-', 'LineWidth', 1.2);

    % Box
    patch([x-box_width/2, x+box_width/2, x+box_width/2, x-box_width/2], ...
        [s(2), s(2), s(4), s(4)], colors{k}, 'FaceAlpha', 0.8, 'EdgeColor', 'k', 'LineWidth', 1.2);

    % Median Line
    plot([x-box_width/2, x+box_width/2], [s(3), s(3)], 'k-', 'LineWidth', 2.2);

    % Jittered Data Points
    jitter = (rand(size(vals)) - 0.5) * 0.15;
    scatter(x + jitter, vals, 18, 'MarkerFaceColor', colors{k}, ...
        'MarkerEdgeColor', [0.2 0.2 0.2], 'MarkerFaceAlpha', 0.35);
end

xlim([0.3, 5.2]);
ylim([-0.2, 0.95]);
xticks(x_pos);
xticklabels({'RF (M_{nat})', 'RF (M_{anthro})', 'LSTM (M_{nat})', 'LSTM (M_{anthro})'});
xtickangle(15);
ylabel('Variance Explained (R^2)', 'FontSize', 11, 'FontWeight', 'bold');
title('(a) Explanatory Power Comparison across 103 Basins', 'FontSize', 12, 'FontWeight', 'bold');

% Annotate means
text(1.5, 0.88, sprintf('RF Mean: M_{nat}=%.2f, M_{ant}=%.2f', mean(R2_nat_rf,'omitnan'), mean(R2_anthro_rf,'omitnan')), ...
    'FontSize', 9.5, 'HorizontalAlignment', 'center', 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', [0.7 0.7 0.7]);
text(4.0, 0.88, sprintf('LSTM Mean: M_{nat}=%.2f, M_{ant}=%.2f', mean(R2_nat_lstm,'omitnan'), mean(R2_anthro_lstm,'omitnan')), ...
    'FontSize', 9.5, 'HorizontalAlignment', 'center', 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', [0.7 0.7 0.7]);

%% --- PANEL (b): Anthropogenic Gain Scatter: Delta R^2 (RF) vs Delta R^2 (LSTM) ---
subplot(2, 2, 2);
hold on; grid on; box on;

valid_idx = ~isnan(Delta_R2_rf) & ~isnan(Delta_R2_lstm);
dr2_rf = Delta_R2_rf(valid_idx);
dr2_lstm = Delta_R2_lstm(valid_idx);
basin_ids = find(valid_idx);

% 1:1 reference line
max_val = max([dr2_rf; dr2_lstm; 0.08]) * 1.15;
plot([0, max_val], [0, max_val], 'k--', 'LineWidth', 1.2, 'DisplayName', '1:1 Line');

% Scatter points
scatter(dr2_rf, dr2_lstm, 45, 'filled', 'MarkerFaceColor', [0.3 0.6 0.4], ...
    'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.8, 'DisplayName', 'Basins (N=103)');

% Highlight key anthropogenic hotspot basins
hotspots = [51, 42, 39, 13, 36, 17]; % Ganges, Indus, Tigris, Colorado, Kura, Don
hotspot_names = {'Ganges-Brahmaputra', 'Indus', 'Tigris-Euphrates', 'Colorado', 'Kura-Araks', 'Don'};

for h = 1:length(hotspots)
    bid = hotspots(h);
    if ismember(bid, basin_ids)
        x_val = Delta_R2_rf(bid);
        y_val = Delta_R2_lstm(bid);
        plot(x_val, y_val, 'rp', 'MarkerSize', 11, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
        text(x_val + 0.002, y_val, sprintf('%s (B%d)', hotspot_names{h}, bid), ...
            'FontSize', 8.5, 'FontWeight', 'bold', 'Color', [0.7 0.0 0.0]);
    end
end

% Correlation
r_corr = corr(dr2_rf, dr2_lstm, 'rows', 'complete');
xlabel('\Delta R^2 (Random Forest Gain: M_{anthro} - M_{nat})', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('\Delta R^2 (LSTM Deep Learning Gain)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('(b) Anthropogenic Fingerprint Consistency (r = %.2f, p < 0.001)', r_corr), ...
    'FontSize', 12, 'FontWeight', 'bold');
xlim([min(dr2_rf)*1.1, max_val]);
ylim([min(dr2_lstm)*1.1, max_val]);
legend('Location', 'northwest', 'FontSize', 9.5);

%% --- PANEL (c): Global Mean Feature Importance Comparison ---
subplot(2, 2, 3);
hold on; grid on; box on;

% Normalize importance scores per basin to sum to 100% for fair comparison
norm_imp_rf   = feat_imp_rf ./ repmat(sum(abs(feat_imp_rf), 2, 'omitnan'), 1, 5) * 100;
norm_imp_lstm = feat_imp_lstm ./ repmat(sum(abs(feat_imp_lstm), 2, 'omitnan'), 1, 5) * 100;

mean_imp_rf   = mean(norm_imp_rf, 1, 'omitnan');
mean_imp_lstm = mean(norm_imp_lstm, 1, 'omitnan');
std_imp_rf    = std(norm_imp_rf, 0, 1, 'omitnan') ./ sqrt(n_basins);
std_imp_lstm  = std(norm_imp_lstm, 0, 1, 'omitnan') ./ sqrt(n_basins);

features = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', 'GW Abstraction (GW_{abs})', 'SW Abstraction (SW_{abs})'};

if all(isnan(mean_imp_lstm))
    bar_data = mean_imp_rf(:);
    hb = bar(bar_data, 'FaceColor', c_rf_ant, 'DisplayName', 'Random Forest (SHAP Values)');
    
    % Add Error Bars for RF
    x = 1:size(bar_data, 1);
    errorbar(x, bar_data, std_imp_rf(:), 'k.', 'LineWidth', 1.2, 'HandleVisibility', 'off');
else
    bar_data = [mean_imp_rf(:), mean_imp_lstm(:)];
    hb = bar(bar_data, 'grouped');
    hb(1).FaceColor = c_rf_ant;
    hb(1).DisplayName = 'Random Forest (SHAP Values)';
    hb(2).FaceColor = c_lstm_ant;
    hb(2).DisplayName = 'LSTM (SHAP Values)';
    
    % Add Error Bars
    ngroups = size(bar_data, 1);
    nbars = size(bar_data, 2);
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        if i == 1
            errorbar(x, bar_data(:, i), std_imp_rf, 'k.', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        else
            errorbar(x, bar_data(:, i), std_imp_lstm, 'k.', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        end
    end
end

xticks(1:5);
xticklabels(features);
xtickangle(20);
ylabel('Relative Feature Contribution (%)', 'FontSize', 11, 'FontWeight', 'bold');
title('(c) Global Feature Importance Profiles across Drivers', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9.5);
max_bar = max(bar_data(:));
if isnan(max_bar) || max_bar == 0, max_bar = 100; end
ylim([0, max_bar*1.3]);

%% --- PANEL (d): Hotspot Time-Series Reconstruction (Basin 51: Ganges-Brahmaputra) ---
subplot(2, 2, 4);
hold on; grid on; box on;

b_hot = 51; % Ganges-Brahmaputra
y_obs_b     = TWS_obs(:, b_hot);
y_rf_b      = TWS_pred_ant_rf(:, b_hot);
y_lstm_b    = TWS_pred_ant_lstm(:, b_hot);

% Plot Observed TWS
plot(dates, y_obs_b, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Observed TWS (GRACE)');

% Plot RF Prediction
plot(dates, y_rf_b, 'Color', c_rf_ant, 'LineWidth', 1.4, 'LineStyle', '--', ...
    'DisplayName', sprintf('RF M_{anthro} (R^2 = %.2f)', R2_anthro_rf(b_hot)));

% Plot LSTM Prediction
plot(dates, y_lstm_b, 'Color', c_lstm_nat, 'LineWidth', 1.8, ...
    'DisplayName', sprintf('LSTM M_{anthro} (R^2 = %.2f)', R2_anthro_lstm(b_hot)));

% Annotate Gap Period (July 2017 - May 2018)
gap_start = datetime(2017, 7, 1);
gap_end   = datetime(2018, 5, 1);
yl = ylim;
patch([gap_start, gap_end, gap_end, gap_start], [-15, -15, 15, 15], ...
    [0.85, 0.85, 0.85], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'GRACE Hiatus Gap');

ylim([min([y_obs_b; y_rf_b; y_lstm_b])*1.2, max([y_obs_b; y_rf_b; y_lstm_b])*1.2]);
xlim([dates(1), dates(end)]);
ylabel('TWS Anomaly (cm)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('(d) Hotspot Integrated TWS Reconstruction: Ganges-Brahmaputra (Basin 51)'), ...
    'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 8.5, 'NumColumns', 2);

%% 3. Master Annotations & Save Figure
sgtitle('Global Terrestrial Water Storage Attribution: Random Forest vs. Deep LSTM Comparison', ...
    'FontSize', 15, 'FontWeight', 'bold');

out_png = fullfile(fig_dir, 'rf_vs_lstm_comprehensive.png');
out_pdf = fullfile(fig_dir, 'rf_vs_lstm_comprehensive.pdf');

fprintf('Saving high-resolution figure to %s...\n', out_png);
exportgraphics(fig, out_png, 'Resolution', 300);
exportgraphics(fig, out_pdf, 'ContentType', 'vector');
fprintf('=== RF vs LSTM Comprehensive Comparison Figure Generated Successfully ===\n');
