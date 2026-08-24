%% plot_shap_summary.m - Visualization of SHAP Feature Attribution
% =========================================================================
% PURPOSE:
%   Generates SHAP summary plots for the top hotspot basins to show the
%   directional impact of abstraction on TWS changes.
% =========================================================================

fprintf('=== Plotting SHAP Summary ===\n');

%% Directory Setup
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
output_dir   = fullfile(project_root, 'outputs', 'figures');
table_dir    = fullfile(project_root, 'outputs', 'tables');

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% Load Data
shap_file = fullfile(table_dir, 'shap_results.mat');
if ~exist(shap_file, 'file')
    error('SHAP results not found. Run src/modeling/step04g_rf_shap_analysis.m first.');
end
load(shap_file, 'shap_results', 'target_basins');

feature_names = {'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', ...
                 'Groundwater (GW_{abs})', 'Surface Water (SW_{abs})'};

%% Generate Plots
for i = 1:length(shap_results)
    if isempty(shap_results{i}), continue; end
    
    res = shap_results{i};
    b = res.basin_id;
    
    % Prepare data
    shap_vals = res.shap_values; % N x 5
    feature_vals = res.X_ant_val; % N x 5
    
    fig = figure('Position', [100, 100, 800, 600], 'Visible', 'off');
    
    % We will create a custom SHAP summary plot
    % Scatter plot for each feature: Y-axis is feature, X-axis is SHAP value,
    % Color is feature value magnitude
    
    hold on;
    num_features = size(shap_vals, 2);
    
    % Create custom colormap (blue to red)
    cmap = parula(256);
    
    for f = 1:num_features
        % Normalize feature values for coloring (0 to 1)
        f_vals = feature_vals(:, f);
        f_norm = (f_vals - min(f_vals)) ./ (max(f_vals) - min(f_vals) + eps);
        
        % Jitter Y for visibility
        y_pos = (num_features - f + 1) + (rand(size(f_vals)) - 0.5) * 0.4;
        
        scatter(shap_vals(:, f), y_pos, 15, f_norm, 'filled', 'MarkerEdgeAlpha', 0.5);
    end
    
    plot([0 0], [0.5 num_features+0.5], 'k--', 'LineWidth', 1.5);
    hold off;
    
    colormap(cmap);
    cb = colorbar;
    cb.Label.String = 'Feature Value (Low to High)';
    cb.Ticks = [0 1];
    cb.TickLabels = {'Low', 'High'};
    
    set(gca, 'YTick', 1:num_features, 'YTickLabel', flip(feature_names));
    ylim([0.5 num_features+0.5]);
    xlabel('SHAP Value (Impact on TWSC Prediction, cm/month)');
    title(sprintf('SHAP Summary Plot - Basin %d', b));
    grid on;
    
    % Save figure
    out_name = fullfile(output_dir, sprintf('shap_summary_basin_%d.png', b));
    saveas(fig, out_name);
    out_name_pdf = fullfile(output_dir, sprintf('shap_summary_basin_%d.pdf', b));
    exportgraphics(fig, out_name_pdf, 'ContentType', 'vector');
    
    fprintf('Saved SHAP plot for Basin %d\n', b);
    close(fig);
end

fprintf('SHAP plotting complete.\n');
