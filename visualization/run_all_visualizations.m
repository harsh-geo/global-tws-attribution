%% run_all_visualizations.m - Master Visualization Runner
clear; clc; close all;
fprintf('=================================================================\n');
fprintf('  GLOBAL TWS ATTRIBUTION - RUNNING ALL VISUALIZATION SCRIPTS     \n');
fprintf('=================================================================\n\n');

scripts = {
    'plot_basin_trends.m'
    'plot_top20_negative_trends.m'
    'plot_delta_r2_lollipop.m'
    'plot_global_attribution_map.m'
    'plot_model_comparison_boxplots.m'
    'plot_rf_vs_lstm_comprehensive.m'
    'plot_rf_vs_lstm_multi_basin_timeseries.m'
    'plot_bootstrap_uncertainty.m'
    'plot_shap_summary.m'
    'plot_spatial_transferability.m'
    'plot_basin_51_analysis.m'
};

vis_dir = fileparts(mfilename('fullpath'));

for i = 1:length(scripts)
    s_name = scripts{i};
    s_path = fullfile(vis_dir, s_name);
    fprintf('\n--> [%d/%d] Running %s...\n', i, length(scripts), s_name);
    try
        run(s_path);
        close all;
        fprintf('✓ %s completed successfully.\n', s_name);
    catch ME
        fprintf('[WARNING] %s failed with error: %s\n', s_name, ME.message);
    end
end

fprintf('\n=================================================================\n');
fprintf('  ALL VISUALIZATION SCRIPTS COMPLETED                            \n');
fprintf('=================================================================\n');
