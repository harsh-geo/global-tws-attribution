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
load(trans_file, 'pristine_basins', 'irrigated_basins', 'bias_results', 'TWSC_pred_transfer');

% Load basin names
proc_dir = fullfile(project_root, 'data', 'processed');
basin_names_mat = fullfile(proc_dir, 'tws_basins.mat');
if exist(basin_names_mat, 'file')
    load(basin_names_mat, 'basin_names');
else
    basin_names = arrayfun(@(b) sprintf('Basin %d', b), 1:103, 'UniformOutput', false);
end

% Load observed TWSC to compute monthly bias distribution & uncertainty
attr_file = fullfile(table_dir, 'attribution_results.mat');
if exist(attr_file, 'file')
    attr_data = load(attr_file, 'TWSC_obs');
    TWSC_obs = attr_data.TWSC_obs;
else
    TWSC_obs = [];
end

%% Generate Bar Chart for Top Biased Basins
fig1 = figure('Position', [100, 100, 950, 600], 'Color', 'w', 'Visible', 'off');

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

% Compute Uncertainty (Standard Error of the Mean Bias via Block-Bootstrap)
top_se = zeros(1, num_to_plot);

if ~isempty(TWSC_obs) && exist('TWSC_pred_transfer', 'var')
    rng(42);
    block_len = 36;
    N_boot = 1000;
    n_time = size(TWSC_obs, 1);
    num_blocks = ceil(n_time / block_len);
    max_start = max(1, n_time - block_len + 1);
    
    target_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';
    baseline_idx = year(target_dates) >= 2004 & year(target_dates) <= 2009;
    
    for i = 1:num_to_plot
        b = top_basins(i);
        
        % Deseasonalize observed TWSC consistently with step04f
        x_obs = TWSC_obs(:, b);
        valid_x = ~isnan(x_obs);
        if sum(valid_x) >= 24
            x_filled = x_obs;
            if any(~valid_x), x_filled = fillmissing(x_obs, 'linear'); end
            [LT, ST_out, R] = trenddecomp(x_filled, 'stl', 12);
            temp_twsc = LT + R;
            base_m = mean(temp_twsc(baseline_idx), 'omitnan');
            twsc_target = temp_twsc - base_m;
            twsc_target(~valid_x) = NaN;
        else
            twsc_target = x_obs;
        end
        
        diff_t = TWSC_pred_transfer(:, b) - twsc_target;
        valid_t = ~isnan(diff_t);
        if sum(valid_t) > 20
            boot_means = zeros(N_boot, 1);
            for k = 1:N_boot
                b_idx = [];
                for blk = 1:num_blocks
                    st = randi(max_start);
                    b_idx = [b_idx; (st:min(n_time, st+block_len-1))'];
                end
                b_idx = b_idx(1:min(n_time, length(b_idx)));
                boot_means(k) = mean(diff_t(b_idx), 'omitnan');
            end
            top_se(i) = std(boot_means);
        end
    end
end

hold on;
% Draw bars
for i = 1:num_to_plot
    bar(i, top_bias(i), 'FaceColor', [0.8500 0.3250 0.0980], 'EdgeColor', 'k', 'LineWidth', 0.5);
    
    % Add errorbar (+/- 1 Standard Error)
    errorbar(i, top_bias(i), top_se(i), 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);
end

% Add zero reference line
plot([0, num_to_plot+1], [0, 0], 'k--', 'LineWidth', 1.2);

% Formatting
xlim([0.5, num_to_plot + 0.5]);
y_min = min([0, min(top_bias - top_se)]) - 0.02;
y_max = max(top_bias + top_se) + 0.03;
ylim([y_min, y_max]);

set(gca, 'XTick', 1:num_to_plot);
top_labels = cell(1, num_to_plot);
for i = 1:num_to_plot
    b = top_basins(i);
    top_labels{i} = basin_names{b}; % Basin name only, without number
end
set(gca, 'XTickLabel', top_labels, 'TickLabelInterpreter', 'none');
xtickangle(45);
ylabel('Mean Prediction Bias (M_{nat} - Observed TWSC, cm/month)', 'FontSize', 11, 'FontWeight', 'bold');
title('Unmodeled Anthropogenic Sink (Top 20 Irrigated Basins, \pm1 SE Uncertainty)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

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
