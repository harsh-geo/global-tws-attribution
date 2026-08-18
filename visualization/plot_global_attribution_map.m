%% plot_global_attribution_map.m
% =========================================================================
% PURPOSE: Create global map of dominant TWS drivers based on Random Forest
%          feature importance.
% =========================================================================

% 1. Set paths and load data
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

raw_dir = fullfile(project_root, 'data', 'raw');
table_dir = fullfile(project_root, 'outputs', 'tables');

basin_mat = fullfile(raw_dir, 'basin_map.mat');
if ~exist(basin_mat, 'file')
    basin_mat = fullfile(project_root, 'data', 'processed', 'basin_map.mat');
end
if ~exist(basin_mat, 'file')
    error('basin_map.mat not found. Please ensure data is available.');
end

basin_data = load(basin_mat);
if isfield(basin_data, 'basin_map')
    basin_mask = double(basin_data.basin_map);
elseif isfield(basin_data, 'basins')
    basin_mask = double(basin_data.basins);
else
    fn = fieldnames(basin_data);
    basin_mask = double(basin_data.(fn{1}));
end

attr_mat = fullfile(table_dir, 'validation_and_trends.mat');
if ~exist(attr_mat, 'file')
    error('File validation_and_trends.mat not found. Run attribution and validation steps first.');
end
load(attr_mat, 'feature_importance');

% feature_importance is [n_basins x 5]: 1:P, 2:ET, 3:Q, 4:GW_abs, 5:SW_abs
n_basins = size(feature_importance, 1);

% Identify dominant driver (max feature importance)
[~, dom_driver] = max(feature_importance, [], 2);

% 2. Reconstruct spatial map
[dim1, dim2] = size(basin_mask);
dom_map = nan(dim1, dim2);

for b = 1:n_basins
    idx = (basin_mask == b);
    dom_map(idx) = dom_driver(b);
end

if dim1 == 720 && dim2 == 360
    dom_map = dom_map';
end

% 3. Create the figure
figure('Name', 'Dominant TWS Drivers', 'Color', 'w', 'Position', [150 150 1000 500]);
h = imagesc(dom_map);
set(gca, 'Color', [0.8 0.8 0.8]); % Gray background
h.AlphaData = ~isnan(dom_map);
axis image;
axis off;

% Custom Categorical Colormap for 5 drivers
% 1: P (Blue), 2: ET (Green), 3: Q (Cyan), 4: GW (Orange), 5: SW (Red)
cmap = [
    0      0.4470 0.7410;  % P - Blue
    0.4660 0.6740 0.1880;  % ET - Green
    0.3010 0.7450 0.9330;  % Q - Cyan
    0.8500 0.3250 0.0980;  % GW - Orange
    0.6350 0.0780 0.1840   % SW - Red
];
colormap(cmap);
caxis([0.5 5.5]);

% Custom colorbar
cb = colorbar;
cb.Ticks = 1:5;
cb.TickLabels = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', 'GW Abstraction', 'SW Abstraction'};
cb.FontSize = 11;
title('Dominant Driver of TWS Trends (Random Forest Feature Importance)', 'FontSize', 14);

% 4. Save figure
out_dir = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
saveas(gcf, fullfile(out_dir, 'tws_dominant_drivers.png'));
fprintf('Saved dominant driver map to %s\n', fullfile(out_dir, 'tws_dominant_drivers.png'));
