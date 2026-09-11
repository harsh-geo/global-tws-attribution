%% plot_global_attribution_map.m
% =========================================================================
% PURPOSE: Create a global choropleth map coloring each basin by its
%          dominant driver of TWS variability (determined by maximum OOB
%          permutation feature importance from M_anthro).
% =========================================================================

% 1. Set paths
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

n_basins = 103;
n_features = 5;

driver_labels = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', ...
                 'Groundwater Abstraction (GW)', 'Surface Water Abstraction (SW)'};

% Publication Color Palette
cmap = [
    0.00, 0.45, 0.74;   % 1: Blue (P)
    0.18, 0.65, 0.24;   % 2: Green (ET)
    0.00, 0.75, 0.85;   % 3: Cyan (Q)
    0.85, 0.33, 0.10;   % 4: Orange (GW_abs)
    0.85, 0.00, 0.00    % 5: Red (SW_abs)
];

% 2. Load Dominant Drivers
dom_driver_idx = nan(n_basins, 1);
s1_csv = fullfile(table_dir, 'Table_S1_Supplementary.csv');
boot_csv = fullfile(table_dir, 'bootstrap_attribution_uncertainty.csv');

if exist(s1_csv, 'file')
    opts = detectImportOptions(s1_csv);
    t_s1 = readtable(s1_csv, opts);
    for b = 1:height(t_s1)
        b_id = t_s1.Basin_ID(b);
        if b_id < 1 || b_id > n_basins, continue; end
        drv = string(t_s1.Dominant_Driver{b});
        switch drv
            case 'P',      dom_driver_idx(b_id) = 1;
            case 'ET',     dom_driver_idx(b_id) = 2;
            case 'Q',      dom_driver_idx(b_id) = 3;
            case 'GW_abs', dom_driver_idx(b_id) = 4;
            case 'SW_abs', dom_driver_idx(b_id) = 5;
            otherwise,     dom_driver_idx(b_id) = 1; % fallback
        end
    end
elseif exist(boot_csv, 'file')
    t_boot = readtable(boot_csv);
    for b = 1:height(t_boot)
        b_id = t_boot.Basin_ID(b);
        if b_id < 1 || b_id > n_basins, continue; end
        imp_vals = [t_boot.Imp_P_Mean(b), t_boot.Imp_ET_Mean(b), t_boot.Imp_Q_Mean(b), ...
                    t_boot.Imp_GW_Mean(b), t_boot.Imp_SW_Mean(b)];
        [~, max_idx] = max(imp_vals);
        dom_driver_idx(b_id) = max_idx;
    end
else
    error('Neither Table_S1_Supplementary.csv nor bootstrap_attribution_uncertainty.csv found.');
end

% 3. Load Basin Mask
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

[dim1, dim2] = size(basin_mask);
if dim1 == 720 && dim2 == 360
    basin_mask = basin_mask';
end

% 4. Create Spatial Driver Grid
driver_map = zeros(size(basin_mask));
for b = 1:n_basins
    idx = (basin_mask == b);
    if ~isnan(dom_driver_idx(b))
        driver_map(idx) = dom_driver_idx(b);
    end
end

% 5. Visualization
fig = figure('Name', 'Global Attribution Map', 'Color', 'w', 'Position', [100, 100, 1280, 680], 'Visible', 'off');
ax = axes('Position', [0.04, 0.18, 0.92, 0.76]);

% Base grid coordinates
lon = linspace(-179.75, 179.75, 720);
lat = linspace(-89.75, 89.75, 360);

% Render driver choropleth
h = imagesc(lon, lat, driver_map);
set(gca, 'YDir', 'normal');
set(gca, 'Color', [0.93 0.95 0.98]); % Soft ocean background
h.AlphaData = (driver_map > 0);
axis image;
axis off;

% Overlay global coastlines
hold on;
try
    load coastlines
    plot(coastlon, coastlat, 'Color', [0.15 0.15 0.15], 'LineWidth', 0.85);
catch
    warning('Could not load coastlines.');
end

colormap(ax, cmap);
caxis([0.5, n_features + 0.5]);

title('Dominant Driver of TWS Variability (M_{anthro} Feature Importance)', ...
    'FontSize', 15, 'FontWeight', 'bold');

% Categorical legend with driver counts
counts = histcounts(dom_driver_idx(dom_driver_idx >= 1 & dom_driver_idx <= 5), 0.5:1:5.5);
legend_labels = cell(n_features, 1);
for i = 1:n_features
    legend_labels{i} = sprintf('%s (n = %d)', driver_labels{i}, counts(i));
end

h_leg = gobjects(n_features, 1);
for i = 1:n_features
    h_leg(i) = patch(NaN, NaN, cmap(i, :), 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.5);
end
L = legend(h_leg, legend_labels, 'Orientation', 'horizontal', 'NumColumns', 5);
L.Position = [0.06, 0.05, 0.88, 0.06];
L.FontSize = 9.5;
L.FontWeight = 'bold';
L.EdgeColor = 'none';

% Save High-Resolution Figures
out_file_png = fullfile(out_dir, 'global_attribution_map.png');
out_file_pdf = fullfile(out_dir, 'global_attribution_map.pdf');
exportgraphics(fig, out_file_png, 'Resolution', 300);
exportgraphics(fig, out_file_pdf, 'ContentType', 'vector');
close(fig);

fprintf('[SUCCESS] Global Attribution Map regenerated and saved to:\n  PNG: %s\n  PDF: %s\n', ...
    out_file_png, out_file_pdf);
