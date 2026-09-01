
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

rf_shap_mat = fullfile(table_dir, 'shap_results.mat');
if exist(rf_shap_mat, 'file')
    shap_data = load(rf_shap_mat);
    shap_res = shap_data.shap_results;
    n_basins = 103;
    feature_importance = nan(n_basins, 5); % Fallback to 5 features
    for i = 1:length(shap_res)
        if ~isempty(shap_res{i})
            feature_importance(i, :) = mean(abs(shap_res{i}.shap_values), 1, 'omitnan');
        end
    end
else
    % Fallback to OOB if SHAP not yet computed for all basins
    trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
    attr_mat  = fullfile(table_dir, 'attribution_results.mat');

    if exist(trend_mat, 'file')
        load(trend_mat, 'feature_importance');
    elseif exist(attr_mat, 'file')
        attr_data = load(attr_mat, 'feature_importance');
        if isfield(attr_data, 'feature_importance') && ~isempty(attr_data.feature_importance)
            feature_importance = attr_data.feature_importance;
        end
    else
        error('Neither SHAP results nor OOB results found.');
    end
end

% Load Delta R2 to determine if anthropogenic impact is significant
attr_mat = fullfile(table_dir, 'attribution_results.mat');
if exist(attr_mat, 'file')
    load(attr_mat, 'Delta_R2');
else
    Delta_R2 = zeros(103, 1);
end

% Ensure feature_importance is [n_basins x n_features]
[dim_a, dim_b] = size(feature_importance);
if dim_a < dim_b && (dim_a == 5 || dim_a == 7)
    feature_importance = feature_importance';
end
[n_basins, n_features] = size(feature_importance);

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

% 2. Determine dominant driver per basin and configure palette
dom_driver_idx = nan(n_basins, 1);
delta_r2_threshold = 0.00; % Threshold for anthropogenic dominance

if n_features == 7
    % 7-Feature Architecture: P, ET, Q, T, ONI, GW_abs, SW_abs
    gw_idx = 6; sw_idx = 7;
    nat_indices = 1:5;
    
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
    gw_idx = 4; sw_idx = 5;
    nat_indices = 1:3;
    
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

for b = 1:n_basins
    if Delta_R2(b) > delta_r2_threshold
        % Anthropogenically driven: compare GW vs SW
        if feature_importance(b, gw_idx) >= feature_importance(b, sw_idx)
            dom_driver_idx(b) = gw_idx;
        else
            dom_driver_idx(b) = sw_idx;
        end
    else
        % Climate driven: max among natural variables
        [~, max_nat] = max(feature_importance(b, nat_indices));
        dom_driver_idx(b) = nat_indices(max_nat);
    end
end

% Map driver index to spatial grid (0 = background, 1..n_features = drivers)
driver_map = zeros(size(basin_mask));

for b = 1:n_basins
    idx = (basin_mask == b);
    if ~isnan(dom_driver_idx(b))
        driver_map(idx) = dom_driver_idx(b);
    end
end

% 3. Visualization
figure('Name', 'Global Attribution Map', 'Color', 'w', 'Position', [100 100 1100 550]);

% Create spatial coordinates for 0.5 deg grid
lon = linspace(-179.75, 179.75, 720);
lat = linspace(-89.75, 89.75, 360);

h = imagesc(lon, lat, driver_map);
set(gca, 'YDir', 'normal'); % Fix inverted map
set(gca, 'Color', [0.85 0.85 0.85]); % Light gray background for non-basin areas
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

colormap(cmap);
caxis([0.5, n_features + 0.5]); % Center colors on integers 1..n_features

title('Dominant Driver of TWS Variability (M_{anthro} Feature Importance)', ...
    'FontSize', 15, 'FontWeight', 'bold');

% Add categorical legend
hold on;
h_leg = gobjects(n_features, 1);
for i = 1:n_features
    h_leg(i) = patch(NaN, NaN, cmap(i, :), 'EdgeColor', 'k');
end
L = legend(h_leg, driver_labels, 'Location', 'southoutside', 'Orientation', 'horizontal', ...
    'NumColumns', min(4, n_features));
L.FontSize = 10.5;
L.EdgeColor = 'none';

% Save figure
out_file = fullfile(out_dir, 'global_attribution_map.png');
saveas(gcf, out_file);
fprintf('Saved Global Attribution Map to %s\n', out_file);
