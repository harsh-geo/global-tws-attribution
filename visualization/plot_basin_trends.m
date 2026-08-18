%% plot_basin_trends.m
% =========================================================================
% PURPOSE: Create global maps of TWS decline trends (cm/year) and 
%          shade/stipple significant basins based on Mann-Kendall test.
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

trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
if ~exist(trend_mat, 'file')
    error('File validation_and_trends.mat not found. Run step05 first.');
end
load(trend_mat, 'tws_trend_slope', 'mk_h_sig');

n_basins = length(tws_trend_slope);

% 2. Reconstruct spatial maps
[dim1, dim2] = size(basin_mask);
trend_map = nan(dim1, dim2);
sig_map = false(dim1, dim2);

for b = 1:n_basins
    idx = (basin_mask == b);
    trend_map(idx) = tws_trend_slope(b);
    sig_map(idx) = mk_h_sig(b);
end

% Standardize orientation to Lat x Lon for display if it is Lon x Lat
if dim1 == 720 && dim2 == 360
    trend_map = trend_map';
    sig_map = sig_map';
end

% 3. Create the figure
figure('Name', 'Global TWS Trends', 'Color', 'w', 'Position', [100 100 1000 500]);
h = imagesc(trend_map);
set(gca, 'Color', [0.8 0.8 0.8]); % Gray background for non-basin areas
h.AlphaData = ~isnan(trend_map);
axis image;
axis off;

% Custom Blue-Red Colormap (Red for decline/negative, Blue for increase/positive)
n = 64;
r = [linspace(1, 1, n), linspace(1, 0, n)]';
g = [linspace(0, 1, n), linspace(1, 0, n)]';
b = [linspace(0, 1, n), linspace(1, 1, n)]';
cmap = [r, g, b];
colormap(cmap);

% Symmetrical color axis around zero
max_val = max(abs(tws_trend_slope(~isnan(tws_trend_slope))));
if isempty(max_val) || max_val == 0
    max_val = 1;
end
caxis([-max_val max_val]);
cb = colorbar;
cb.Label.String = 'TWS Trend (cm/year)';
cb.FontSize = 12;

% Overlay stippling for non-significant basins
hold on;
non_sig_map = ~sig_map & ~isnan(trend_map);
[row, col] = find(non_sig_map);
% Subsample for stippling so it's not completely dense
stipple_step = 4;
plot(col(1:stipple_step:end), row(1:stipple_step:end), 'k.', 'MarkerSize', 2, 'Color', [0.4 0.4 0.4]);

title('Global TWS Trends (Stippled = Not Significant at p < 0.05)', 'FontSize', 14);

% 4. Save figure
out_dir = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
saveas(gcf, fullfile(out_dir, 'tws_basin_trends.png'));
fprintf('Saved TWS trend map to %s\n', fullfile(out_dir, 'tws_basin_trends.png'));
