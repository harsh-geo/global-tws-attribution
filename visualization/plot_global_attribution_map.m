%% plot_global_attribution_map.m
% =========================================================================
% PURPOSE: Create a global choropleth map coloring each basin by its 
%          dominant driver of TWS variability (determined by maximum OOB 
%          permutation feature importance from M_anthro).
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
load(trend_mat, 'feature_importance');

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

% Standardize orientation to Lat x Lon for display
[dim1, dim2] = size(basin_mask);
if dim1 == 720 && dim2 == 360
    basin_mask = basin_mask';
end

n_basins = size(feature_importance, 1);

% 2. Determine dominant driver per basin
% Feature order: 1:P, 2:ET, 3:Q, 4:GW_abs, 5:SW_abs
[~, dom_driver_idx] = max(feature_importance, [], 2);

% Map driver index to spatial grid
% 0 = background, 1-5 = drivers
driver_map = zeros(size(basin_mask));

for b = 1:n_basins
    idx = (basin_mask == b);
    if ~isnan(dom_driver_idx(b))
        driver_map(idx) = dom_driver_idx(b);
    end
end

% 3. Visualization
figure('Name', 'Global Attribution Map', 'Color', 'w', 'Position', [100 100 1000 500]);

% Create spatial coordinates for 0.5 deg grid
lon = linspace(-179.75, 179.75, 720);
lat = linspace(-89.75, 89.75, 360);

h = imagesc(lon, lat, driver_map);
set(gca, 'YDir', 'normal'); % Fix inverted map
set(gca, 'Color', [0.8 0.8 0.8]); % Gray background for non-basin areas
h.AlphaData = (driver_map > 0);
axis image;
axis off;

% Overlay coastlines
hold on;
try
    load coastlines
    plot(coastlon, coastlat, 'k-', 'LineWidth', 0.8);
catch
    warning('Could not load coastlines.');
end

% Custom categorical colormap
% 1=P (Blue), 2=ET (Green), 3=Q (Cyan), 4=GW (Orange), 5=SW (Red)
cmap = [
    0, 0, 1;         % Blue (P)
    0, 0.8, 0;       % Green (ET)
    0, 1, 1;         % Cyan (Q)
    1, 0.5, 0;       % Orange (GW_abs)
    1, 0, 0          % Red (SW_abs)
];
colormap(cmap);
caxis([0.5, 5.5]); % Center colors on integers 1-5

title('Dominant Driver of TWS Variability', 'FontSize', 16, 'FontWeight', 'bold');

% Add categorical legend
driver_labels = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', ...
                 'Groundwater Abstraction (GW)', 'Surface Water Abstraction (SW)'};

hold on;
for i = 1:5
    h_leg(i) = patch(NaN, NaN, cmap(i,:), 'EdgeColor', 'k');
end
L = legend(h_leg, driver_labels, 'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 3);
L.FontSize = 11;
L.EdgeColor = 'none';

% Save figure
out_file = fullfile(out_dir, 'global_attribution_map.png');
saveas(gcf, out_file);
fprintf('Saved Global Attribution Map to %s\n', out_file);
