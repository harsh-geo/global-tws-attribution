%% plot_delta_r2_lollipop.m
% =========================================================================
% PURPOSE: Create a sorted bar chart of Variance Explained Gain (Delta R^2)
%          for the Top 10 and Bottom 5 basins. 
%          Bars are colored by dominant driver category.
% =========================================================================

% 1. Set paths and load data
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

proc_dir = fullfile(project_root, 'data', 'processed');
table_dir = fullfile(project_root, 'outputs', 'tables');
out_dir = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
attr_mat  = fullfile(table_dir, 'attribution_results.mat');

if exist(trend_mat, 'file')
    load(trend_mat, 'Delta_R2', 'feature_importance');
elseif exist(attr_mat, 'file')
    attr_data = load(attr_mat, 'Delta_R2', 'feature_importance');
    if isfield(attr_data, 'feature_importance') && ~isempty(attr_data.feature_importance)
        feature_importance = attr_data.feature_importance;
        Delta_R2 = attr_data.Delta_R2;
    end
else
    error('Neither validation_and_trends.mat nor attribution_results.mat found.');
end

% Ensure feature_importance is [n_basins x n_features]
[dim_a, dim_b] = size(feature_importance);
if dim_a < dim_b && (dim_a == 5 || dim_a == 7)
    feature_importance = feature_importance';
end

basin_names_mat = fullfile(proc_dir, 'tws_basins.mat');
if ~exist(basin_names_mat, 'file')
    error('File tws_basins.mat not found.');
end
load(basin_names_mat, 'basin_names');

n_basins = length(Delta_R2);
n_features = size(feature_importance, 2);

% 2. Determine dominant drivers and configure palette
[~, dom_driver_idx] = max(feature_importance, [], 2);

if n_features == 7
    % 7-Feature Architecture: P, ET, Q, T, ONI, GW_abs, SW_abs
    driver_labels = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', ...
                     'Temperature (T)', 'ENSO (ONI)', ...
                     'Groundwater Abstraction (GW)', 'Surface Water Abstraction (SW)'};
    cmap = [
        0.00, 0.45, 0.74;   % 1: Blue (P)
        0.18, 0.65, 0.24;   % 2: Green (ET)
        0.00, 0.75, 0.85;   % 3: Cyan (Q)
        0.93, 0.69, 0.13;   % 4: Gold/Amber (T / t2m)
        0.60, 0.20, 0.75;   % 5: Purple (ONI)
        0.85, 0.33, 0.10;   % 6: Orange (GW_abs)
        0.85, 0.00, 0.00    % 7: Red (SW_abs)
    ];
else
    % 5-Feature Architecture: P, ET, Q, GW_abs, SW_abs
    driver_labels = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', ...
                     'Groundwater Abstraction (GW)', 'Surface Water Abstraction (SW)'};
    cmap = [
        0.00, 0.45, 0.74;   % 1: Blue (P)
        0.18, 0.65, 0.24;   % 2: Green (ET)
        0.00, 0.75, 0.85;   % 3: Cyan (Q)
        0.85, 0.33, 0.10;   % 4: Orange (GW_abs)
        0.85, 0.00, 0.00    % 5: Red (SW_abs)
    ];
end

% 3. Sort data by Delta_R2 descending
[sorted_delta_full, sort_idx] = sort(Delta_R2, 'descend');
sorted_names_full = basin_names(sort_idx);
sorted_drivers_full = dom_driver_idx(sort_idx);

% Subset to top 10 and bottom 5
subset_idx = [1:10, (n_basins-4):n_basins];
sorted_delta = sorted_delta_full(subset_idx);
sorted_names = sorted_names_full(subset_idx);
sorted_drivers = sorted_drivers_full(subset_idx);
n_subset = length(subset_idx);

% Pre-assign bar colors
bar_colors = zeros(n_subset, 3);
for i = 1:n_subset
    if ~isnan(sorted_drivers(i))
        bar_colors(i, :) = cmap(sorted_drivers(i), :);
    else
        bar_colors(i, :) = [0.7 0.7 0.7]; % Gray if missing
    end
end

% 4. Create the Bar Chart
figure('Name', 'Delta R2 Attribution', 'Color', 'w', 'Position', [100 100 1050 600]);
hold on;

% Draw bars
b = bar(1:n_subset, sorted_delta, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.5);
b.CData = bar_colors;

% Add zero reference line
plot([0, n_subset+1], [0 0], 'k--', 'LineWidth', 1.5);

% Formatting
xlim([0, n_subset+1]);
ylim([min(sorted_delta)-0.05, max(sorted_delta)+0.1]);

xticks(1:n_subset);
xticklabels(sorted_names);
xtickangle(45);
set(gca, 'TickLabelInterpreter', 'none');

ylabel('Variance Explained Gain (\DeltaR^2 = R^2_{anthro} - R^2_{nat})', 'FontSize', 12, 'FontWeight', 'bold');
title('Improvement in TWS Modeling with Anthropogenic Predictors (Top 10 & Bottom 5 Basins)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
box on;

% Add categorical legend (only for drivers that exist in the palette)
h_leg = gobjects(n_features, 1);
for i = 1:n_features
    h_leg(i) = patch(NaN, NaN, cmap(i, :), 'EdgeColor', 'k');
end
L = legend(h_leg, driver_labels, 'Location', 'northeast');
L.FontSize = 10;

% Save figure
out_file = fullfile(out_dir, 'delta_r2_attribution_barchart.png');
saveas(gcf, out_file);
fprintf('Saved Delta R2 Bar Chart to %s\n', out_file);
