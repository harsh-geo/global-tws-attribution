%% plot_spatial_transferability.m - Visualization of Transferability Test
% =========================================================================
% PURPOSE:
%   Generates a bias map showing where the M_nat model (trained on Pristine
%   basins) systematically overpredicts TWS in Irrigated basins, proving 
%   the causal magnitude of human depletion.
% =========================================================================

fprintf('=== Plotting Spatial Transferability Results ===\n');

%% Directory Setup
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
output_dir   = fullfile(project_root, 'outputs', 'figures');
table_dir    = fullfile(project_root, 'outputs', 'tables');
data_dir     = fullfile(project_root, 'data', 'raw'); % Assuming basin mask is here or similar

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% Load Data
trans_file = fullfile(table_dir, 'transferability_results.mat');
if ~exist(trans_file, 'file')
    error('Transferability results not found. Run src/modeling/step04f_spatial_transferability.m first.');
end
load(trans_file, 'pristine_basins', 'irrigated_basins', 'bias_results');

% bias_results: Positive means model overpredicts TWS (missing anthropogenic sink)
% We want to visualize this bias.

%% Generate Bar Chart for Top Biased Basins
fig1 = figure('Position', [100, 100, 800, 600], 'Visible', 'off');

% Sort irrigated basins by bias
[sorted_bias, sort_idx] = sort(bias_results(irrigated_basins), 'descend', 'MissingPlacement', 'last');
sorted_basins = irrigated_basins(sort_idx);

% Remove NaNs
valid_idx = ~isnan(sorted_bias);
sorted_bias = sorted_bias(valid_idx);
sorted_basins = sorted_basins(valid_idx);

num_to_plot = min(20, length(sorted_bias));
top_bias = sorted_bias(1:num_to_plot);
top_basins = sorted_basins(1:num_to_plot);

bar(1:num_to_plot, top_bias, 'FaceColor', [0.8500 0.3250 0.0980]);
set(gca, 'XTick', 1:num_to_plot, 'XTickLabel', top_basins);
xtickangle(45);
xlabel('Basin ID');
ylabel('Mean Prediction Bias (M_{nat} - Observed TWSC, cm/month)');
title('Unmodeled Anthropogenic Sink (Top 20 Irrigated Basins)');
grid on;

% Save figure
out_name1 = fullfile(output_dir, 'transferability_bias_barchart.png');
saveas(fig1, out_name1);
out_name_pdf1 = fullfile(output_dir, 'transferability_bias_barchart.pdf');
exportgraphics(fig1, out_name_pdf1, 'ContentType', 'vector');

fprintf('Saved Transferability bar chart.\n');
close(fig1);

%% Note: To generate a global map, you would load the basin 2D mask 
% and assign bias_results to each basin ID, similar to plot_basin_trends.m.
% For brevity, we focus on the quantitative bar chart here.

fprintf('Spatial Transferability plotting complete.\n');
