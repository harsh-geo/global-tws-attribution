%% step02c_append_new_drivers.m - Append T_basin and ONI_index to time series
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% =========================================================================

fprintf('=== STEP 2c: Appending Temperature & ENSO Drivers ===\n');

%% Directory Setup
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
processed_dir = fullfile(project_root, 'data', 'processed');
raw_dir       = fullfile(project_root, 'data', 'raw');

%% Load existing basin time-series
ts_file = fullfile(processed_dir, 'basin_time_series.mat');
if ~exist(ts_file, 'file')
    error('Basin time series not found. Run step02_aggregate_basins.m first.');
end
load(ts_file); % Loads P_basin, ET_basin, Q_basin, GW_basin, SW_basin, TWS_basin, grace_dates

n_time = length(grace_dates);
n_basins = size(P_basin, 2);

%% Load Basin Mask & Weights
basin_mat = fullfile(processed_dir, 'basin_map.mat');
if ~exist(basin_mat, 'file')
    basin_mat = fullfile(raw_dir, 'basin_map.mat');
end
basin_data = load(basin_mat);
if isfield(basin_data, 'basin_map')
    basin_mask = double(basin_data.basin_map);
elseif isfield(basin_data, 'basins')
    basin_mask = double(basin_data.basins);
else
    fn = fieldnames(basin_data);
    basin_mask = double(basin_data.(fn{1}));
end

lat_coords = -89.75:0.5:89.75;
cos_weights_1d = cosd(lat_coords);
cos_weights_2d = repmat(cos_weights_1d, 720, 1);

%% 1. Process ONI (ENSO Index)
% The oni.data file format: Year Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
% -99.9 signifies missing value
oni_file = fullfile(raw_dir, 'oni.data');
if ~exist(oni_file, 'file')
    error('oni.data not found. Run download_oni.sh first.');
end

fprintf('Processing ONI data...\n');
fid = fopen(oni_file, 'r');
oni_raw = textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f', 'MultipleDelimsAsOne', true);
fclose(fid);

oni_years = oni_raw{1};
oni_matrix = [oni_raw{2:13}];
ONI_index = nan(n_time, 1);

for t = 1:n_time
    yr = year(grace_dates(t));
    mo = month(grace_dates(t));
    
    row_idx = find(oni_years == yr, 1);
    if ~isempty(row_idx)
        val = oni_matrix(row_idx, mo);
        if val ~= -99.9
            ONI_index(t) = val;
        end
    end
end
fprintf('Extracted %d months of ONI data.\n', sum(~isnan(ONI_index)));

%% 2. Process ERA5 Temperature
t2m_file = fullfile(processed_dir, 'era5_t2m_05deg.nc');
if ~exist(t2m_file, 'file')
    warning('era5_t2m_05deg.nc not found. T_basin will be set to zeros. Run CDO preprocessing first if you want Temperature.');
    T_basin = zeros(n_time, n_basins);
else
    fprintf('Processing ERA5 Temperature (T2M)...\n');
    t2m_info = ncinfo(t2m_file);
    t2m_varname = '';
    for i = 1:length(t2m_info.Variables)
        if strcmp(t2m_info.Variables(i).Name, 't2m')
            t2m_varname = 't2m';
            break;
        end
    end
    if isempty(t2m_varname)
        error('Variable t2m not found in NetCDF.');
    end
    
    T_grid = ncread(t2m_file, t2m_varname);
    
    % Ensure T_grid is [720 x 360 x 216] (216 = 12 months * 18 years from 2002 to 2019)
    % We need to match with grace_dates (starts Apr 2002). Apr 2002 is index 4.
    if size(T_grid, 3) < n_time + 3
        error('T2M grid does not have enough time steps.');
    end
    
    T_grid_aligned = T_grid(:, :, 4:(3+n_time)); % Extract Apr 2002 to Dec 2019
    
    T_basin = extract_weighted_basin_series(T_grid_aligned, basin_mask, cos_weights_2d, n_basins);
    clear T_grid T_grid_aligned;
end

%% Save Updated Time Series
fprintf('Saving appended basin time-series to %s...\n', ts_file);
save(ts_file, 'P_basin', 'ET_basin', 'Q_basin', 'GW_basin', 'SW_basin', 'TWS_basin', 'grace_dates', 'T_basin', 'ONI_index', '-v7.3');
fprintf('=== STEP 2c Complete ===\n');

%% Helper Function
function basin_ts = extract_weighted_basin_series(grid_3d, basin_mask, weight_2d, n_basins)
    [~, ~, n_time] = size(grid_3d);
    basin_ts = nan(n_time, n_basins);
    
    for b = 1:n_basins
        idx = (basin_mask == b);
        if ~any(idx(:))
            continue;
        end
        
        w = weight_2d(idx);
        w_sum = sum(w);
        
        for t = 1:n_time
            slice = grid_3d(:, :, t);
            vals = slice(idx);
            
            % Remove NaNs
            valid = ~isnan(vals);
            if any(valid)
                basin_ts(t, b) = sum(vals(valid) .* w(valid)) / sum(w(valid));
            end
        end
    end
end
