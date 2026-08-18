%% plot_top20_negative_trends.m
% =========================================================================
% PURPOSE: Create a vertical bar chart of the 20 basins with the most 
%          negative TWS trend (in km3/year).
%          Significance is indicated by bar color.
% =========================================================================

% 1. Set paths and load data
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

raw_dir = fullfile(project_root, 'data', 'raw');
proc_dir = fullfile(project_root, 'data', 'processed');
table_dir = fullfile(project_root, 'outputs', 'tables');
out_dir = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
if ~exist(trend_mat, 'file')
    error('File validation_and_trends.mat not found.');
end
load(trend_mat, 'tws_trend_slope', 'mk_h_sig');

basin_names_mat = fullfile(proc_dir, 'tws_basins.mat');
if ~exist(basin_names_mat, 'file')
    error('File tws_basins.mat not found.');
end
load(basin_names_mat, 'basin_names');

basin_map_mat = fullfile(proc_dir, 'basin_map.mat');
if ~exist(basin_map_mat, 'file')
    basin_map_mat = fullfile(raw_dir, 'basin_map.mat');
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

% 2. Compute basin areas and volume trends
[dim1, dim2] = size(basin_mask);
if dim1 == 720 && dim2 == 360
    basin_mask = basin_mask';
end

% Area of each 0.5 deg grid cell in km2
% Cell area = (111.32 * 0.5) * (111.32 * cosd(lat) * 0.5)
lat_vec = linspace(-89.75, 89.75, 360)';
cell_area_map = repmat((111.32 * 0.5)^2 * cosd(lat_vec), 1, 720);

n_basins = length(tws_trend_slope);
basin_area = nan(1, n_basins);
for b = 1:n_basins
    basin_area(b) = sum(cell_area_map(basin_mask == b));
end

% Calculate trend in km3/year
% cm/year * 1e-5 km/cm * km^2 = km^3/year
trend_km3_yr = tws_trend_slope .* 1e-5 .* basin_area;

% 3. Find the 20 most negative trends
[sorted_trends, sort_idx] = sort(trend_km3_yr, 'ascend');
top20_idx = sort_idx(1:20);
top20_trends = sorted_trends(1:20);
top20_names = basin_names(top20_idx);
top20_sig = mk_h_sig(top20_idx);

% 4. Create the bar chart
figure('Name', 'Top 20 Negative Trends', 'Color', 'w', 'Position', [100, 100, 1000, 500]);
hold on;

% Plot bars individually to set colors based on significance
for i = 1:20
    if top20_sig(i) == 1
        % Significant: Red color
        bar_color = [0.8500, 0.3250, 0.0980];
    else
        % Not significant: Gray color
        bar_color = [0.7 0.7 0.7];
    end
    bar(i, top20_trends(i), 'FaceColor', bar_color, 'EdgeColor', 'k');
end

% Formatting
xticks(1:20);
xticklabels(top20_names);
xtickangle(45);
xlim([0.5, 20.5]);
ylabel('TWS Trend (km^3/year)', 'FontSize', 12, 'FontWeight', 'bold');
title('Top 20 Basins with Most Negative TWS Trends', 'FontSize', 14);
grid on;
box on;

% Add custom legend for significance
h_sig = bar(NaN, NaN, 'FaceColor', [0.8500, 0.3250, 0.0980], 'EdgeColor', 'k');
h_nsig = bar(NaN, NaN, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
legend([h_sig, h_nsig], {'Significant (p < 0.05)', 'Not Significant'}, 'Location', 'northeast');

% Save figure
saveas(gcf, fullfile(out_dir, 'top20_negative_trends.png'));
fprintf('Saved Top 20 Negative Trends Bar Chart to %s\n', fullfile(out_dir, 'top20_negative_trends.png'));
