%% step01_unit_conversion.m - Hydroclimate Flux Standardization & NaN Sanitization
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% GOVERNING MASS BALANCE: dTWS/dt = TWSC = P - ET - Q - (GW_abs + SW_abs)
% =========================================================================
% PURPOSE:
%   1. Ingest datasets (GRACE from 'data/raw/', all other hydroclimate
%      datasets from 'data/processed/')
%   2. Replace non-physical fill values (-999, values > 1e19) with MATLAB NaNs
%   3. Standardize all hydroclimate fluxes (P, ET, Q, GW_abs, SW_abs) and
%      GRACE TWS anomalies to liquid water height equivalent depth (cm/month).
%
% INPUT DATASETS:
%   - data/raw/grace_corrected_monthly.nc         -> GRACE TWS (lwe_thickness)
%   - data/processed/precip_era5_monthly_2002_2019.nc  -> ERA5 Precipitation ('tp')
%   - data/processed/et_gleam_global_2002_2021.nc      -> GLEAM Evapotranspiration ('E')
%   - data/processed/runoff_era5_monthly_2002_2019.nc  -> ERA5 Runoff/Discharge ('ro'/'dis')
%   - data/processed/gw_abs_pcrglobwb_2002_2019.nc     -> PCR-GLOBWB GW Abstraction
%   - data/processed/sw_abs_pcrglobwb_2002_2019.nc     -> PCR-GLOBWB SW Abstraction
%
% OUTPUT:
%   - data/processed/standardized_grids.mat containing converted 3D grids
% =========================================================================

clear; clc;
fprintf('=== STEP 1: Starting Unit Conversion & Data Sanitization ===\n');

%% Directory Configuration & Setup
raw_dir       = fullfile('data', 'raw');
processed_dir = fullfile('data', 'processed');
grace_dir     = fullfile('grace');

if ~exist(processed_dir, 'dir')
    mkdir(processed_dir);
    fprintf('Created output directory: %s\n', processed_dir);
end

%% Helper Lambda Functions for Data Sanitization
% Replaces non-physical fill values (-999, values > 1e19) with standard MATLAB NaN
clean_nans = @(x) double(replace_fill_values(x));

%% 1. Process GRACE / GRACE-FO TWS Anomalies (from grace/)
% -------------------------------------------------------------------------
% GRACE TWS is stored as Liquid Water Equivalent (LWE) thickness (in cm) hence no unit conversion is needed.
grace_file = fullfile(grace_dir, 'grace_corrected_monthly.nc');
if ~exist(grace_file, 'file')
    grace_file = fullfile(raw_dir, 'grace_corrected_monthly.nc');
end

if exist(grace_file, 'file')
    fprintf('[1/6] Loading GRACE TWS anomalies from %s...\n', grace_file);
    tws_raw = ncread(grace_file, 'lwe_thickness');

    tws_raw = clean_nans(tws_raw);

    tws_grid = tws_raw; % Already in cm
    clear tws_raw;
else
    warning('GRACE file %s not found. Proceeding assuming downstream dummy placeholder.', grace_file);
    tws_grid = [];
end

%% 2. Process ERA5 Precipitation (P) (from data/processed/)
% -------------------------------------------------------------------------
% ERA5 monthly total precipitation ('tp') is provided in meters/month depth.
% Target unit: cm/month -> multiply by 100.
p_file = fullfile(processed_dir, 'precip_era5_monthly_2002_2019.nc');
if exist(p_file, 'file')
    fprintf('[2/6] Loading ERA5 Precipitation from %s...\n', p_file);
    p_raw = ncread(p_file, 'tp');
    p_raw = clean_nans(p_raw);

    % Conversion: m/month -> cm/month (x 100)
    P_grid = p_raw * 100.0;
    fprintf('   Precipitation converted: m/month -> cm/month (scaled x100).\n');
    clear p_raw;
else
    warning('Precipitation file %s not found.', p_file);
    P_grid = [];
end

%% 3. Process GLEAM Evapotranspiration (ET) (from data/processed/)
% -------------------------------------------------------------------------
% GLEAM monthly evapotranspiration ('E') is provided in mm/month.
% Target unit: cm/month -> divide mm by 10.
et_file = fullfile(processed_dir, 'et_gleam_global_2002_2021.nc');
if exist(et_file, 'file')
    fprintf('[3/6] Loading GLEAM Evapotranspiration from %s...\n', et_file);
    et_raw = ncread(et_file, 'E');
    et_raw = clean_nans(et_raw);

    % Conversion: mm/month -> cm/month (/ 10)
    ET_grid = et_raw / 10.0;
    fprintf('   Evapotranspiration converted: mm/month -> cm/month (scaled /10).\n');
    clear et_raw;
else
    warning('Evapotranspiration file %s not found.', et_file);
    ET_grid = [];
end

%% 4. Process ERA5 Runoff / Discharge (Q) (from data/processed/)
% -------------------------------------------------------------------------
% ERA5 Total Runoff ('ro') is provided in meters/month depth.
% Target unit: cm/month -> multiply by 100.
q_file = fullfile(processed_dir, 'runoff_era5_monthly_2002_2019.nc');
if ~exist(q_file, 'file')
    q_file = fullfile(processed_dir, 'runoff_era5_monthly_2002_2021.nc');
end

if exist(q_file, 'file')
    fprintf('[4/6] Loading ERA5 Runoff/Discharge from %s...\n', q_file);
    q_raw = ncread(q_file, 'ro');
    q_raw = clean_nans(q_raw);

    % Convert to cm/month (x 100)
    Q_grid = q_raw * 100.0;
    fprintf('   Runoff converted: m/month -> cm/month (scaled x100).\n');
    clear q_raw;
else
    warning('Runoff file %s not found.', q_file);
    Q_grid = [];
end

%% 5. Process PCR-GLOBWB Groundwater Abstraction (GW_abs) (from data/processed/)
% -------------------------------------------------------------------------
% PCR-GLOBWB groundwater abstraction is provided in m/month depth.
% Target unit: cm/month.
gw_file = fullfile(processed_dir, 'gw_abs_pcrglobwb_2002_2019.nc');
if exist(gw_file, 'file')
    fprintf('[5/6] Loading PCR-GLOBWB GW Abstraction from %s...\n', gw_file);
    gw_raw = ncread(gw_file, 'total_groundwater_abstraction');
    gw_raw = clean_nans(gw_raw);

    GW_grid = gw_raw * 100.0; % m/month -> cm/month
    fprintf('   GW Abstraction converted to cm/month.\n');
    clear gw_raw;
else
    warning('GW Abstraction file %s not found.', gw_file);
    GW_grid = [];
end

%% 6. Process PCR-GLOBWB Surface Water Abstraction (SW_abs) (from data/processed/)
% -------------------------------------------------------------------------
% PCR-GLOBWB surface water abstraction is provided in m/month depth.
% Target unit: cm/month.
sw_file = fullfile(processed_dir, 'sw_abs_pcrglobwb_2002_2019.nc');
if exist(sw_file, 'file')
    fprintf('[6/6] Loading PCR-GLOBWB SW Abstraction from %s...\n', sw_file);
    sw_raw = ncread(sw_file, 'surface_water_abstraction');
    sw_raw = clean_nans(sw_raw);

    SW_grid = sw_raw * 100.0; % m/month -> cm/month
    fprintf('   SW Abstraction converted to cm/month.\n');
    clear sw_raw;
else
    warning('SW Abstraction file %s not found.', sw_file);
    SW_grid = [];
end

%% Save Standardized Grids
out_mat = fullfile(processed_dir, 'standardized_grids.mat');
fprintf('Saving converted spatial grids to %s...\n', out_mat);
save(out_mat, 'tws_grid', 'P_grid', 'ET_grid', 'Q_grid', 'GW_grid', 'SW_grid', '-v7.3');
fprintf('=== STEP 1 Complete: All Fluxes Standardized to cm/month ===\n\n');

%% Local Helper Function for NaN Replacement
function data = replace_fill_values(data)
% Replaces non-physical fill values (-999, values > 1e19, -9999) with NaN
data(data <= -900) = NaN;
data(data > 1e19)  = NaN;
data(isinf(data))  = NaN;
end
