%% plot_study_domain.m - Study Domain Map of 103 Major Global River Basins
% =========================================================================
% Generates Figure 1: Global map showing the 103 river basin polygons
% with continent/ID color shading and major basin labels.
% =========================================================================

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

raw_dir   = fullfile(project_root, 'data', 'raw');
out_dir   = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

basin_mat = fullfile(raw_dir, 'basin_map.mat');
if ~exist(basin_mat, 'file')
    basin_mat = fullfile(project_root, 'data', 'processed', 'basin_map.mat');
end
if ~exist(basin_mat, 'file')
    error('basin_map.mat not found.');
end

load(basin_mat);
if exist('basin_map', 'var')
    mask = double(basin_map);
elseif exist('basins', 'var')
    mask = double(basins);
else
    fn = fieldnames(basin_data);
    mask = double(basin_data.(fn{1}));
end

if size(mask, 1) == 720 && size(mask, 2) == 360
    mask = mask';
end

load(fullfile(project_root, 'data', 'processed', 'tws_basins.mat'), 'basin_names');

fig = figure('Position', [100, 100, 1100, 580], 'Color', 'w', 'Visible', 'off');
ax = axes('Position', [0.05, 0.08, 0.90, 0.85]);

% Custom discrete map of basins
cmap = jet(103);
% Shuffle colormap to make neighboring basins distinct
rng(42);
cmap = cmap(randperm(103), :);

imagesc([-179.75, 179.75], [-89.75, 89.75], mask);
set(gca, 'YDir', 'normal');
colormap(ax, cmap);
caxis([1, 103]);

% Shading for ocean / non-basin areas
hold on;
ocean_mask = (mask <= 0 | isnan(mask));
h_nan = imagesc([-179.75, 179.75], [-89.75, 89.75], double(ocean_mask));
set(h_nan, 'AlphaData', double(ocean_mask) * 0.95);
colormap(ax, [0.93 0.95 0.98; cmap]); % Very light gray-blue for oceans

% Grid and labels
grid on;
set(gca, 'GridColor', [0.7 0.7 0.7], 'GridAlpha', 0.4, 'LineWidth', 0.8);
xlim([-180, 180]);
ylim([-60, 85]);
xlabel('Longitude (\circ)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Latitude (\circ)', 'FontSize', 11, 'FontWeight', 'bold');
title('Global River Basins Domain (103 Major Basins, 0.5^\circ \times 0.5^\circ)', ...
    'FontSize', 14, 'FontWeight', 'bold');

% Key basin labels
labels = {
    'Amazon', -62, -4;
    'Congo', 22, -1;
    'Mississippi', -90, 38;
    'Ganges-Brahmaputra', 86, 26;
    'Indus', 71, 31;
    'Tigris-Euphrates', 44, 34;
    'Yukon', -150, 64;
    'Mackenzie', -124, 63;
    'Volga', 46, 55;
    'Danube', 21, 46;
    'Yangtze', 112, 31;
    'Murray-Darling', 144, -32;
    'Parana', -58, -26;
    'Nile', 31, 15
};

for i = 1:size(labels, 1)
    name = labels{i, 1};
    lon = labels{i, 2};
    lat = labels{i, 3};
    text(lon, lat, name, 'FontSize', 8, 'FontWeight', 'bold', ...
        'Color', [0.05, 0.05, 0.05], 'BackgroundColor', [1 1 1 0.75], ...
        'EdgeColor', [0.4 0.4 0.4], 'Margin', 1.5, 'HorizontalAlignment', 'center');
end

set(gca, 'FontSize', 10, 'TickDir', 'out', 'Box', 'on');

% Save Figure
exportgraphics(fig, fullfile(out_dir, 'study_domain_map.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(out_dir, 'study_domain_map.pdf'), 'ContentType', 'vector');
close(fig);
fprintf('✓ Study domain map saved to %s\n', fullfile(out_dir, 'study_domain_map.png'));
