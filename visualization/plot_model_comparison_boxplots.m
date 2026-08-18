%% plot_model_comparison_boxplots.m
% =========================================================================
% PURPOSE: Create box-and-whisker plots comparing performance metrics 
%          (NSE, KGE, RMSE) between Natural (M_nat) and Anthropogenic 
%          (M_anthro) models across all 103 basins.
%          Whiskers extend to 5th and 95th percentiles.
% =========================================================================

% 1. Set paths and load data
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

table_dir = fullfile(project_root, 'outputs', 'tables');
out_dir = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

trend_mat = fullfile(table_dir, 'validation_and_trends.mat');
if ~exist(trend_mat, 'file')
    error('File validation_and_trends.mat not found.');
end
load(trend_mat, 'NSE_nat', 'NSE_anthro', 'KGE_nat', 'KGE_anthro', ...
    'RMSE_cv_nat', 'RMSE_cv_anthro');

% 2. Prepare data for plotting
metrics = {'NSE', 'KGE', 'RMSE'};
data_nat = {NSE_nat, KGE_nat, RMSE_cv_nat};
data_anthro = {NSE_anthro, KGE_anthro, RMSE_cv_anthro};

% Colors
color_nat = [0, 0.4470, 0.7410]; % Blue
color_anthro = [0.8500, 0.3250, 0.0980]; % Orange

figure('Name', 'Model Comparison Boxplots', 'Color', 'w', 'Position', [100, 100, 1000, 400]);

% Function to calculate custom boxplot stats (5th, 25th, 50th, 75th, 95th)
calc_stats = @(x) [prctile(x, 5), prctile(x, 25), prctile(x, 50), prctile(x, 75), prctile(x, 95)];

% We will use a custom drawing loop to enforce exact 5th and 95th percentiles
for m = 1:3
    subplot(1, 3, m);
    hold on;
    
    val_nat = data_nat{m}(:);
    val_anthro = data_anthro{m}(:);
    
    % Remove NaNs
    val_nat = val_nat(~isnan(val_nat));
    val_anthro = val_anthro(~isnan(val_anthro));
    
    stats_nat = calc_stats(val_nat);
    stats_anthro = calc_stats(val_anthro);
    
    x_centers = [1, 2];
    box_width = 0.4;
    
    % --- Plot M_nat (Blue) ---
    x = x_centers(1);
    s = stats_nat;
    % Whiskers (5 to 25 and 75 to 95)
    plot([x x], [s(1) s(2)], 'k-', 'LineWidth', 1.2);
    plot([x x], [s(4) s(5)], 'k-', 'LineWidth', 1.2);
    % Whisker caps
    plot([x-0.1 x+0.1], [s(1) s(1)], 'k-', 'LineWidth', 1.2);
    plot([x-0.1 x+0.1], [s(5) s(5)], 'k-', 'LineWidth', 1.2);
    % Box (25 to 75)
    patch([x-box_width/2 x+box_width/2 x+box_width/2 x-box_width/2], ...
          [s(2) s(2) s(4) s(4)], color_nat, 'FaceAlpha', 0.8, 'EdgeColor', 'k', 'LineWidth', 1.2);
    % Median
    plot([x-box_width/2 x+box_width/2], [s(3) s(3)], 'k-', 'LineWidth', 2);
    
    % Outliers
    outliers_nat = val_nat(val_nat < s(1) | val_nat > s(5));
    if ~isempty(outliers_nat)
        plot(repmat(x, size(outliers_nat)), outliers_nat, 'o', ...
            'MarkerEdgeColor', color_nat, 'MarkerFaceColor', color_nat, 'MarkerSize', 4);
    end
    
    % --- Plot M_anthro (Orange) ---
    x = x_centers(2);
    s = stats_anthro;
    % Whiskers
    plot([x x], [s(1) s(2)], 'k-', 'LineWidth', 1.2);
    plot([x x], [s(4) s(5)], 'k-', 'LineWidth', 1.2);
    % Whisker caps
    plot([x-0.1 x+0.1], [s(1) s(1)], 'k-', 'LineWidth', 1.2);
    plot([x-0.1 x+0.1], [s(5) s(5)], 'k-', 'LineWidth', 1.2);
    % Box
    patch([x-box_width/2 x+box_width/2 x+box_width/2 x-box_width/2], ...
          [s(2) s(2) s(4) s(4)], color_anthro, 'FaceAlpha', 0.8, 'EdgeColor', 'k', 'LineWidth', 1.2);
    % Median
    plot([x-box_width/2 x+box_width/2], [s(3) s(3)], 'k-', 'LineWidth', 2);
    
    % Outliers
    outliers_anthro = val_anthro(val_anthro < s(1) | val_anthro > s(5));
    if ~isempty(outliers_anthro)
        plot(repmat(x, size(outliers_anthro)), outliers_anthro, 'o', ...
            'MarkerEdgeColor', color_anthro, 'MarkerFaceColor', color_anthro, 'MarkerSize', 4);
    end
    
    % Formatting
    xticks(x_centers);
    xticklabels({'M_{nat}', 'M_{anthro}'});
    xlim([0.2, 2.8]);
    title(metrics{m}, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Metric-specific formatting
    if strcmp(metrics{m}, 'NSE') || strcmp(metrics{m}, 'KGE')
        ylabel('Score', 'FontSize', 12, 'FontWeight', 'bold');
        ylim([-0.5, 1]); % Common scale for NSE/KGE
    else
        ylabel('Error (cm)', 'FontSize', 12, 'FontWeight', 'bold');
        ylim([0, max([val_nat; val_anthro])*1.1]);
    end
    
    grid on;
    box on;
end

% Add a master legend
h_nat = patch(NaN, NaN, color_nat, 'FaceAlpha', 0.8, 'EdgeColor', 'k');
h_anthro = patch(NaN, NaN, color_anthro, 'FaceAlpha', 0.8, 'EdgeColor', 'k');
L = legend([h_nat, h_anthro], {'M_{nat} (Natural Baseline)', 'M_{anthro} (Full Anthropogenic)'}, ...
    'Position', [0.35 0.92 0.3 0.05], 'Orientation', 'horizontal', 'FontSize', 12);
L.Units = 'normalized';
L.EdgeColor = 'none';

% Save figure
out_file = fullfile(out_dir, 'model_comparison_boxplots.png');
saveas(gcf, out_file);
fprintf('Saved Box-and-Whisker Model Comparison Plot to %s\n', out_file);
