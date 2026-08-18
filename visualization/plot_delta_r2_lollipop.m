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
if ~exist(trend_mat, 'file')
    error('File validation_and_trends.mat not found.');
end
load(trend_mat, 'Delta_R2', 'feature_importance');

basin_names_mat = fullfile(proc_dir, 'tws_basins.mat');
if ~exist(basin_names_mat, 'file')
    error('File tws_basins.mat not found.');
end
load(basin_names_mat, 'basin_names');

n_basins = length(Delta_R2);

% 2. Determine dominant drivers
% 1:P, 2:ET, 3:Q, 4:GW_abs, 5:SW_abs
[~, dom_driver_idx] = max(feature_importance, [], 2);

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

% 4. Custom categorical colormap
% 1=P (Blue), 2=ET (Green), 3=Q (Cyan), 4=GW (Orange), 5=SW (Red)
cmap = [
    0, 0, 1;         % Blue (P)
    0, 0.8, 0;       % Green (ET)
    0, 1, 1;         % Cyan (Q)
    1, 0.5, 0;       % Orange (GW_abs)
    1, 0, 0          % Red (SW_abs)
];

% Pre-assign bar colors
bar_colors = zeros(n_subset, 3);
for i = 1:n_subset
    if ~isnan(sorted_drivers(i))
        bar_colors(i, :) = cmap(sorted_drivers(i), :);
    else
        bar_colors(i, :) = [0.7 0.7 0.7]; % Gray if missing
    end
end

% 5. Create the Bar Chart
figure('Name', 'Delta R2 Attribution', 'Color', 'w', 'Position', [100 100 1000 600]);
hold on;

% Draw bars
b = bar(1:n_subset, sorted_delta, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.5);
b.CData = bar_colors;

% Add zero reference line
plot([0, n_subset+1], [0 0], 'k--', 'LineWidth', 1.5);

% 6. Formatting
xlim([0, n_subset+1]);
ylim([min(sorted_delta)-0.05, max(sorted_delta)+0.1]);

xticks(1:n_subset);
xticklabels(sorted_names);
xtickangle(45);
set(gca, 'TickLabelInterpreter', 'none');

ylabel('Variance Explained Gain (\DeltaR^2 = R^2_{anthro} - R^2_{nat})', 'FontSize', 12, 'FontWeight', 'bold');
title('Improvement in TWS Modeling with Anthropogenic Predictors (Top 10 & Bottom 5 Basins)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% 7. Add categorical legend
driver_labels = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', ...
                 'Groundwater Abstraction (GW)', 'Surface Water Abstraction (SW)'};
for i = 1:5
    h_leg(i) = patch(NaN, NaN, cmap(i,:), 'EdgeColor', 'k');
end
L = legend(h_leg, driver_labels, 'Location', 'northeast');
L.FontSize = 11;

% Save figure
out_file = fullfile(out_dir, 'delta_r2_attribution_barchart.png');
saveas(gcf, out_file);
fprintf('Saved Delta R2 Bar Chart to %s\n', out_file);
