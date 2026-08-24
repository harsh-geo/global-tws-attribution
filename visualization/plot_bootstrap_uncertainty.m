%% plot_bootstrap_uncertainty.m - Plot Empirical Bootstrap Confidence Intervals & Uncertainty
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Visualize 95% empirical confidence intervals for Delta R^2 across
%      key depleted river basins.
%   2. Plot feature importance bootstrap uncertainty distributions.
%
% OUTPUT:
%   - outputs/figures/fig_bootstrap_uncertainty.png
%   - outputs/figures/fig_bootstrap_uncertainty.pdf
% =========================================================================

script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'outputs'), 'dir')
    project_root = pwd;
end

table_dir  = fullfile(project_root, 'outputs', 'tables');
figure_dir = fullfile(project_root, 'outputs', 'figures');

if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end
proc_dir = fullfile(project_root, 'data', 'processed');

boot_mat = fullfile(table_dir, 'bootstrap_uncertainty_results.mat');
if ~exist(boot_mat, 'file')
    error('Bootstrap results file not found: %s. Run step04c_bootstrap_uncertainty.m first.', boot_mat);
end

fprintf('Loading bootstrap uncertainty data from %s...\n', boot_mat);
load(boot_mat); %#ok<LOAD>

% Load trends data if available for basin sorting
trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
if exist(trend_mat, 'file')
    t_data = load(trend_mat);
    if isfield(t_data, 'tws_trend_slope')
        tws_slope = t_data.tws_trend_slope;
    else
        tws_slope = zeros(1, size(boot_delta_r2_dist, 2));
    end
else
    tws_slope = zeros(1, size(boot_delta_r2_dist, 2));
end

%% Select Basins for Uncertainty Forest Plot
% Select top 15 basins with strongest anthropogenic gain / depletion
valid_basins = find(~isnan(delta_r2_mean));
[~, sort_order] = sort(delta_r2_mean(valid_basins), 'descend');
top_k = min(15, length(valid_basins));
selected_basins = valid_basins(sort_order(1:top_k));

%% Create Multi-Panel Publication Figure
fig = figure('Color', 'w', 'Position', [100, 100, 1200, 550], 'Visible', 'off');

% --- Subplot 1: Forest Plot of Delta R^2 with 95% Bootstrap CI ---
subplot(1, 2, 1);
hold on;
box on;
grid on;
set(gca, 'FontSize', 10, 'LineWidth', 1.0, 'GridAlpha', 0.15);

y_pos = 1:top_k;
means = delta_r2_mean(selected_basins);
ci_l  = delta_r2_ci_low(selected_basins);
ci_u  = delta_r2_ci_upp(selected_basins);

% Reference line at 0 gain
plot([0, 0], [0, top_k + 1], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

for i = 1:top_k
    % 95% CI Whisker
    line([ci_l(i), ci_u(i)], [y_pos(i), y_pos(i)], 'Color', [0.15 0.35 0.65], 'LineWidth', 2.0);
    
    % Cap markers
    plot(ci_l(i), y_pos(i), '|', 'Color', [0.15 0.35 0.65], 'MarkerSize', 8, 'LineWidth', 1.5);
    plot(ci_u(i), y_pos(i), '|', 'Color', [0.15 0.35 0.65], 'MarkerSize', 8, 'LineWidth', 1.5);
    
    % Point estimate (Mean)
    plot(means(i), y_pos(i), 'o', 'MarkerFaceColor', [0.85 0.25 0.20], ...
        'MarkerEdgeColor', 'k', 'MarkerSize', 7, 'LineWidth', 1.0);
end

ylim([0.5, top_k + 0.5]);
yticks(y_pos);

basin_names_mat = fullfile(proc_dir, 'tws_basins.mat');
if exist(basin_names_mat, 'file')
    b_data = load(basin_names_mat, 'basin_names');
    basin_labels = b_data.basin_names(selected_basins);
else
    basin_labels = arrayfun(@(b) sprintf('Basin %d', b), selected_basins, 'UniformOutput', false);
end
yticklabels(basin_labels);
set(gca, 'TickLabelInterpreter', 'none');
xlabel('Anthropogenic Explanatory Gain (\Delta R^2)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('(a) \\Delta R^2 Attribution Gain (95%% Bootstrap CI, N=%d)', N_boot), ...
    'FontSize', 12, 'FontWeight', 'bold');

% --- Subplot 2: Feature Importance Empirical Distributions ---
subplot(1, 2, 2);
hold on;
box on;
grid on;
set(gca, 'FontSize', 10, 'LineWidth', 1.0, 'GridAlpha', 0.15);

feature_names = {'P', 'ET', 'Q', 'GW_{abs}', 'SW_{abs}'};
n_feat = length(feature_names);

% Aggregate all bootstrap draws across all basins for global feature importance spread
global_imp_dist = squeeze(mean(boot_feat_imp_dist, 2, 'omitnan')); % [N_boot x n_features]

colors = [
    0.20, 0.50, 0.80;  % P - blue
    0.25, 0.70, 0.45;  % ET - green
    0.35, 0.75, 0.85;  % Q - cyan
    0.85, 0.20, 0.20;  % GW_abs - red
    0.90, 0.45, 0.25   % SW_abs - coral
];

boxplot(global_imp_dist, 'Labels', feature_names, 'Colors', 'k', 'Symbol', 'r.', ...
    'Widths', 0.55);

h_boxes = findobj(gca, 'Tag', 'Box');
for j = 1:length(h_boxes)
    patch(get(h_boxes(j), 'XData'), get(h_boxes(j), 'YData'), ...
        colors(n_feat - j + 1, :), 'FaceAlpha', 0.65, 'EdgeColor', 'k', 'LineWidth', 1.0);
end

xlabel('Hydroclimate & Anthropogenic Predictors', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('OOB Permutation Importance (\Delta Error)', 'FontSize', 11, 'FontWeight', 'bold');
title('(b) Global Predictor Importance Bootstrap Distributions', 'FontSize', 12, 'FontWeight', 'bold');

%% Save Figure
out_png = fullfile(figure_dir, 'fig_bootstrap_uncertainty.png');
out_pdf = fullfile(figure_dir, 'fig_bootstrap_uncertainty.pdf');

exportgraphics(fig, out_png, 'Resolution', 300);
exportgraphics(fig, out_pdf, 'ContentType', 'vector');
close(fig);

fprintf('Saved bootstrap uncertainty figure to:\n  - %s\n  - %s\n', out_png, out_pdf);
