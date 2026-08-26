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

% 3. Load trend slope and compute volumetric loss to sort basins
vars_trend = load(trend_mat, 'tws_trend_slope');
tws_trend_slope = vars_trend.tws_trend_slope;

basin_map_mat = fullfile(proc_dir, 'basin_map.mat');
if ~exist(basin_map_mat, 'file')
    basin_map_mat = fullfile(project_root, 'data', 'raw', 'basin_map.mat');
end
basin_data = load(basin_map_mat);
if isfield(basin_data, 'basin_map')
    basin_mask = double(basin_data.basin_map);
elseif isfield(basin_data, 'basins')
    basin_mask = double(basin_data.basins);
else
    fn = fieldnames(basin_data);
    basin_mask = double(basin_data.(fn{1}));
end

[dim1, dim2] = size(basin_mask);
if dim1 == 720 && dim2 == 360
    basin_mask = basin_mask';
end
lat_vec = linspace(-89.75, 89.75, 360)';
cell_area_map = repmat((111.32 * 0.5)^2 * cosd(lat_vec), 1, 720);

basin_area = nan(1, n_basins);
for b = 1:n_basins
    basin_area(b) = sum(cell_area_map(basin_mask == b));
end
trend_km3_yr = tws_trend_slope .* 1e-5 .* basin_area;

% Sort by volumetric loss (most negative trend first)
[~, sort_idx] = sort(trend_km3_yr, 'ascend');

sorted_delta_full = Delta_R2(sort_idx);
sorted_names_full = basin_names(sort_idx);
sorted_drivers_full = dom_driver_idx(sort_idx);

% Choose top 15 basins with most volumetric TWS loss
subset_idx = 1:15;
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
for i = 1:n_subset
    bar(i, sorted_delta(i), 'FaceColor', bar_colors(i, :), 'EdgeColor', 'k', 'LineWidth', 0.5);
end

% Add zero reference line
plot([0, n_subset+1], [0 0], 'k--', 'LineWidth', 1.5);

% Formatting
xlim([0, n_subset+1]);
ylim([min(sorted_delta)-0.05, max(sorted_delta)+0.1]);

xticks(1:n_subset);
short_driver_names_5 = {'P', 'ET', 'Q', 'GW', 'SW'};
short_driver_names_7 = {'P', 'ET', 'Q', 'T', 'ONI', 'GW', 'SW'};
labeled_names = cell(1, n_subset);
for i = 1:n_subset
    if isnan(sorted_drivers(i))
        labeled_names{i} = sorted_names{i};
    else
        if n_features == 7
            short_name = short_driver_names_7{sorted_drivers(i)};
        else
            short_name = short_driver_names_5{sorted_drivers(i)};
        end
        labeled_names{i} = sprintf('%s (%s)', sorted_names{i}, short_name);
    end
end
xticklabels(labeled_names);
xtickangle(45);
set(gca, 'TickLabelInterpreter', 'none');

ylabel('Variance Explained Gain (\DeltaR^2 = R^2_{anthro} - R^2_{nat})', 'FontSize', 12, 'FontWeight', 'bold');
title('Improvement in TWS Modeling with Anthropogenic Predictors (Top 15 Basins by Volumetric Loss)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
box on;

% Add categorical legend (only for drivers that exist in the palette)
h_leg = gobjects(n_features, 1);
for i = 1:n_features
    h_leg(i) = patch(NaN, NaN, 'white', 'FaceColor', cmap(i, :), 'EdgeColor', 'k');
end
L = legend(h_leg, driver_labels, 'Location', 'northeast');
L.FontSize = 10;

% Save figure
out_file = fullfile(out_dir, 'delta_r2_attribution_barchart.png');
saveas(gcf, out_file);
fprintf('Saved Delta R2 Bar Chart to %s\n', out_file);
