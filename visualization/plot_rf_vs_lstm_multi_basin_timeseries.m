%% plot_rf_vs_lstm_multi_basin_timeseries.m
% =========================================================================
% AUTHOR: Computational Hydrology Research
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% PURPOSE: Create a 4-panel comparison of Observed TWSC vs RF vs LSTM across
%          four contrasting global hydroclimatic and human management regimes:
%          1. Basin 51: Ganges-Brahmaputra (Intensive Irrigation & Monsoon)
%          2. Basin 39: Tigris-Euphrates (Arid Aquifer Depletion)
%          3. Basin 1: Amazon (Humid Tropical, Natural Climate Dominance)
%          4. Basin 5: Yukon (High-Latitude Cold Snowmelt & Groundwater Memory)
% =========================================================================

clear; clc; close all;

%% 1. Set Paths and Load Data
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

table_dir = fullfile(project_root, 'outputs', 'tables');
fig_dir   = fullfile(project_root, 'outputs', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

rf_mat   = fullfile(table_dir, 'attribution_results.mat');
lstm_mat = fullfile(table_dir, 'attribution_results_lstm.mat');

rf_data   = load(rf_mat);
lstm_data = load(lstm_mat);

TWS_pred_ant_rf = rf_data.TWS_pred_anthro_all;
R2_anthro_rf     = rf_data.R2_TWS_anthro;
Delta_R2_rf      = rf_data.R2_TWS_anthro - rf_data.R2_TWS_nat;

TWS_pred_ant_lstm   = lstm_data.TWS_pred_anthro_all;
R2_anthro_lstm       = lstm_data.R2_TWS_anthro;
Delta_R2_lstm        = lstm_data.R2_TWS_anthro - lstm_data.R2_TWS_nat;

% Use TWSC bounds for CI but apply them to TWS conceptually, or just hide CI for simplicity in integrated view
TWSC_pred_lstm_upper = []; 
TWSC_pred_lstm_lower = [];

% Load observed TWS for the timeseries plot
ts_mat = fullfile(project_root, 'data', 'processed', 'grace_reconstructed.mat');
ts_data = load(ts_mat);
TWS_obs = ts_data.TWS_reconstructed;

n_time = size(TWS_obs, 1);
dates  = datetime(2002, 4, 1) + calmonths(0:n_time-1);

%% 2. Setup Figure
fig = figure('Name', 'RF vs LSTM 4-Basin Showcase', ...
    'Units', 'inches', 'Position', [1, 1, 14, 11], 'Color', 'w');

basins = [51, 39, 1, 5];
titles = {
    'Ganges-Brahmaputra (Basin 51): Heavy Agricultural Pumping & Monsoon', ...
    'Tigris-Euphrates (Basin 39): Arid Climate & Aquifer Overdraft', ...
    'Amazon (Basin 1): Humid Tropical, Pristine Hydroclimate Dominance', ...
    'Yukon (Basin 5): Cold Snowmelt Catchment & Storage Delay Dynamics'
};

c_obs  = [0.15, 0.15, 0.15];
c_rf   = [0.00, 0.55, 0.85];
c_lstm = [0.85, 0.35, 0.10];
c_ci   = [0.95, 0.75, 0.55];

gap_start = datetime(2017, 7, 1);
gap_end   = datetime(2018, 5, 1);

for i = 1:4
    b = basins(i);
    subplot(2, 2, i);
    hold on; grid on; box on;
    
    y_obs  = TWS_obs(:, b);
    y_rf   = TWS_pred_ant_rf(:, b);
    y_lstm = TWS_pred_ant_lstm(:, b);
    
    % LSTM 95% Confidence Interval Envelope
    if ~isempty(TWSC_pred_lstm_upper)
        y_up = TWSC_pred_lstm_upper(:, b);
        y_lo = TWSC_pred_lstm_lower(:, b);
        valid = ~isnan(y_up) & ~isnan(y_lo);
        fill([dates(valid), fliplr(dates(valid))], [y_up(valid)', fliplr(y_lo(valid)')], ...
            c_ci, 'FaceAlpha', 0.35, 'EdgeColor', 'none', 'DisplayName', 'LSTM 95% CI');
    end
    
    % Observations
    plot(dates, y_obs, 'k-', 'LineWidth', 1.4, 'DisplayName', 'GRACE TWSC Obs');
    
    % RF Model
    plot(dates, y_rf, 'Color', c_rf, 'LineWidth', 1.4, 'LineStyle', '--', ...
        'DisplayName', sprintf('RF (R^2 = %.2f, \\Delta R^2 = +%.3f)', R2_anthro_rf(b), Delta_R2_rf(b)));
    
    % LSTM Model
    plot(dates, y_lstm, 'Color', c_lstm, 'LineWidth', 1.8, ...
        'DisplayName', sprintf('LSTM (R^2 = %.2f, \\Delta R^2 = +%.3f)', R2_anthro_lstm(b), Delta_R2_lstm(b)));
    
    % Shade Inter-mission Gap
    yl = [min([y_obs; y_rf; y_lstm])*1.25, max([y_obs; y_rf; y_lstm])*1.25];
    patch([gap_start, gap_end, gap_end, gap_start], [yl(1), yl(1), yl(2), yl(2)], ...
        [0.85, 0.85, 0.85], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'GRACE Hiatus');
    
    ylim(yl);
    xlim([dates(1), dates(end)]);
    ylabel('Integrated TWS Anomaly (cm)', 'FontSize', 10.5, 'FontWeight', 'bold');
    title(sprintf('(%s) %s', char(96+i), titles{i}), 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'southwest', 'FontSize', 8, 'NumColumns', 2);
end

sgtitle('Hydroclimatic Regime Contrast: Random Forest vs. Recurrent LSTM Storage Predictions', ...
    'FontSize', 14, 'FontWeight', 'bold');

out_png = fullfile(fig_dir, 'rf_vs_lstm_four_basins.png');
out_pdf = fullfile(fig_dir, 'rf_vs_lstm_four_basins.pdf');

fprintf('Saving 4-basin showcase figure to %s...\n', out_png);
exportgraphics(fig, out_png, 'Resolution', 300);
exportgraphics(fig, out_pdf, 'ContentType', 'vector');
fprintf('=== 4-Basin Showcase Figure Generated Successfully ===\n');
