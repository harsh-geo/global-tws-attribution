%% plot_top15_feature_importance.m - Figure 9 Feature Importance Breakdown
% =========================================================================
% Generates Figure 9: Horizontal stacked bar chart of feature importance
% for the 15 most severely declining significant river basins.
% Colors: P (Blue), ET (Green), Q (Cyan), GW_abs (Orange), SW_abs (Red)
% =========================================================================

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

table_dir = fullfile(project_root, 'outputs', 'tables');
out_dir   = fullfile(project_root, 'outputs', 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

sum_csv = fullfile(table_dir, 'basin_summary_table.csv');
unc_csv = fullfile(table_dir, 'attribution_uncertainty.csv');

T_sum = readtable(sum_csv);
T_unc = readtable(unc_csv);

% Filter to significant declining basins
sig_decl = (T_sum.Is_Significant == 1) & (T_sum.Trend_km3_yr < 0);
T_decl = T_sum(sig_decl, :);

% Sort by volumetric rate
T_decl = sortrows(T_decl, 'Trend_km3_yr', 'ascend');
top15 = T_decl(1:min(15, height(T_decl)), :);

% Reverse order so most severe is at the top of horizontal bar chart
top15 = top15(end:-1:1, :);

b_ids = top15.Basin_ID;
b_names = top15.Basin_Name;

% Match importance values
n_top = length(b_ids);
imp_matrix = zeros(n_top, 5); % P, ET, Q, GW, SW

for i = 1:n_top
    id = b_ids(i);
    row_u = T_unc(T_unc.Basin_ID == id, :);
    if ~isempty(row_u)
        raw_imp = [row_u.Imp_P(1), row_u.Imp_ET(1), row_u.Imp_Q(1), row_u.Imp_GW(1), row_u.Imp_SW(1)];
        raw_imp(isnan(raw_imp)) = 0;
        if sum(raw_imp) > 0
            imp_matrix(i, :) = (raw_imp / sum(raw_imp)) * 100;
        else
            imp_matrix(i, :) = [20, 20, 20, 20, 20];
        end
    end
end

% Plot
fig = figure('Position', [100, 100, 950, 620], 'Color', 'w', 'Visible', 'off');
ax = axes('Position', [0.28, 0.10, 0.68, 0.82]);

% Colors: P (Blue), ET (Green), Q (Cyan), GW_abs (Orange), SW_abs (Red)
colors = [
    0.12, 0.47, 0.71;  % P (Blue)
    0.17, 0.63, 0.17;  % ET (Green)
    0.10, 0.74, 0.81;  % Q (Cyan)
    1.00, 0.50, 0.05;  % GW (Orange)
    0.84, 0.15, 0.16   % SW (Red)
];

b_plot = barh(imp_matrix, 'stacked', 'BarWidth', 0.65, 'EdgeColor', 'none');
for c = 1:5
    b_plot(c).FaceColor = colors(c, :);
end

% Y-ticks
set(gca, 'YTick', 1:n_top, 'YTickLabel', b_names, 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Relative Feature Importance (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, 100]);
title('Driver Attribution Breakdown: Top 15 Severely Declining River Basins', ...
    'FontSize', 13, 'FontWeight', 'bold');

grid on;
set(gca, 'XGrid', 'on', 'YGrid', 'off', 'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.6);

lgd = legend({'Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)', ...
              'Groundwater Abstraction (GW_{abs})', 'Surface Water Abstraction (SW_{abs})'}, ...
              'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 9);
lgd.Box = 'off';

% Annotate volumetric rate on the right of each bar
for i = 1:n_top
    rate = top15.Trend_km3_yr(i);
    text(101.5, i, sprintf('%.1f km^3/yr', rate), 'FontSize', 8.5, 'Color', [0.3 0.3 0.3], ...
        'VerticalAlignment', 'middle');
end
xlim([0, 116]);

exportgraphics(fig, fullfile(out_dir, 'fig_feature_importance_top15.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(out_dir, 'fig_feature_importance_top15.pdf'), 'ContentType', 'vector');
close(fig);
fprintf('✓ Feature importance top 15 plot saved to %s\n', fullfile(out_dir, 'fig_feature_importance_top15.png'));
