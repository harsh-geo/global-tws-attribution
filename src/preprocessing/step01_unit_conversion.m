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
%   - Hydroclimate flux grids (tws_grid, P_grid, ET_grid, Q_grid, GW_grid, SW_grid)
%     retained in memory for Step 2 (intermediate .mat saving disabled to save disk space).
% =========================================================================

fprintf('=== STEP 1: Starting Unit Conversion & Data Sanitization ===\n');

%% Directory Configuration & Setup
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

raw_dir       = fullfile(project_root, 'data', 'raw');
processed_dir = fullfile(project_root, 'data', 'processed');
grace_dir     = fullfile(project_root, 'grace');

if ~exist(processed_dir, 'dir')
    mkdir(processed_dir);
    fprintf('Created output directory: %s\n', processed_dir);
end

%% Helper Lambda Functions for Data Sanitization
% Replaces non-physical fill values (-999, values > 1e19) with standard MATLAB NaN
clean_nans = @(x) double(replace_fill_values(x));

%% Target Timeline Setup (Apr 2002 to Dec 2019 = 213 months)
target_dates = (datetime(2002, 4, 1) + calmonths(0:212))'; % 213 x 1 datetime vector
[target_ym_y, target_ym_m] = ymd(target_dates);

%% 1. Process GRACE / GRACE-FO TWS Anomalies (from grace/)
% -------------------------------------------------------------------------
% GRACE TWS is stored as Liquid Water Equivalent (LWE) thickness (in cm).
% GRACE missing months are not stored as NaNs in the file, but are missing
% from the time dimension. We re-index GRACE onto the 213-month timeline (Apr 2002 - Dec 2019)
% placing NaNs at missing months.
grace_file = fullfile(grace_dir, 'grace_corrected_monthly.nc');
dates_file = fullfile(grace_dir, 'grace_dates.mat');

if ~exist(grace_file, 'file')
    grace_file = fullfile(raw_dir, 'grace_corrected_monthly.nc');
end
if ~exist(dates_file, 'file')
    dates_file = fullfile(raw_dir, 'grace_dates.mat');
end

if exist(grace_file, 'file')
    fprintf('[1/6] Loading GRACE TWS anomalies from %s...\n', grace_file);
    tws_raw = ncread(grace_file, 'lwe_thickness');
    tws_raw = clean_nans(tws_raw);

    [n_lon, n_lat, n_grace_time] = size(tws_raw);
    tws_grid = nan(n_lon, n_lat, 213); % Pre-allocate 213-timestep grid with NaNs

    if exist(dates_file, 'file')
        d_struct = load(dates_file);
        fn = fieldnames(d_struct);
        g_dates = d_struct.(fn{1});
        [g_y, g_m] = ymd(g_dates);

        matched_count = 0;
        for k = 1:min(n_grace_time, length(g_dates))
            idx = find(target_ym_y == g_y(k) & target_ym_m == g_m(k));
            if ~isempty(idx)
                tws_grid(:, :, idx(1)) = tws_raw(:, :, k);
                matched_count = matched_count + 1;
            end
        end
        fprintf('   Mapped %d observed GRACE months onto 213-month timeline (Apr 2002 - Dec 2019).\n', matched_count);
    else
        n_copy = min(n_grace_time, 213);
        tws_grid(:, :, 1:n_copy) = tws_raw(:, :, 1:n_copy);
        fprintf('   grace_dates.mat not found. Copied first %d timesteps into timeline.\n', n_copy);
    end
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
    
    % Read time to calculate days in each month
    t_sec = double(ncread(p_file, 'valid_time'));
    dt = datetime(t_sec, 'ConvertFrom', 'posixtime');
    days_in_month = eomday(year(dt), month(dt));
    days_mult = reshape(days_in_month, 1, 1, []);

    % Conversion: m/day -> cm/month (x days_in_month x 100)
    P_grid = p_raw .* days_mult * 100.0;
    fprintf('   Precipitation converted: m/day -> cm/month (scaled by days in month x100).\n');
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

    % Align GLEAM ET grid coordinates to basin map (-180..180 lon, -90..90 lat):
    % 1. Flip latitude dimension (North->South to South->North)
    % 2. Shift longitude by 180 deg (360 grid cells) to align [-180..180] spatial domain
    fprintf('   Aligning GLEAM ET spatial extent (Latitude flip + Longitude 180-deg circshift)...\n');
    et_raw = circshift(flip(et_raw, 2), [360, 0, 0]);

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
    
    % Read time to calculate days in each month
    t_sec_q = double(ncread(q_file, 'valid_time'));
    dt_q = datetime(t_sec_q, 'ConvertFrom', 'posixtime');
    days_in_month_q = eomday(year(dt_q), month(dt_q));
    days_mult_q = reshape(days_in_month_q, 1, 1, []);

    % Convert to cm/month (x days_in_month x 100)
    Q_grid = q_raw .* days_mult_q * 100.0;
    fprintf('   Runoff converted: m/day -> cm/month (scaled by days in month x100).\n');
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

%% Retain Standardized Grids in Memory (Intermediate .mat save disabled to save disk space)
fprintf('=== STEP 1 Complete: All Fluxes Standardized to cm/month (Retained in Memory) ===\n\n');

%% Local Helper Function for NaN Replacement
function data = replace_fill_values(data)
% Replaces non-physical fill values (-999, values > 1e19, -9999) with NaN
data(data <= -900) = NaN;
data(data > 1e19)  = NaN;
data(isinf(data))  = NaN;
end
