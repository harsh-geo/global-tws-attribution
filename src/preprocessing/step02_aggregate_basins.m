%% step02_aggregate_basins.m - Cosine Latitude-Weighted Spatial Aggregation Across 103 Basins
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Load standardized 3D spatio-temporal grids (720x360xN_time) from Step 1.
%   2. Load 103 major global river basin masks from 'data/raw/basin_map.mat'.
%   3. Compute area-weighted (latitude cosine-weighted: cosd(lat)) spatial 
%      averages for each basin across all timesteps.
%   4. Memory Management (GEMINI.md Directive): Explicitly call 'clear' on 
%      heavy 3D spatial grids immediately after basin time-series extraction.
%
% OUTPUT:
%   - data/processed/basin_time_series.mat containing 2D matrices (N_time x 103):
%     P_basin, ET_basin, Q_basin, GW_basin, SW_basin, TWS_basin, grace_dates
% =========================================================================

clear; clc;
fprintf('=== STEP 2: Starting Latitude Cosine-Weighted Basin Aggregation ===\n');

%% Directory Setup & Loading Inputs
processed_dir = fullfile('data', 'processed');
raw_dir       = fullfile('data', 'raw');

grid_mat  = fullfile(processed_dir, 'standardized_grids.mat');
basin_mat = fullfile(raw_dir, 'basin_map.mat');

if ~exist(grid_mat, 'file')
    error('Standardized grids file %s not found! Please run step01_unit_conversion.m first.', grid_mat);
end

if ~exist(basin_mat, 'file')
    error('Basin map file %s not found in data/raw/!', basin_mat);
end

fprintf('Loading basin masks from %s...\n', basin_mat);
basin_data = load(basin_mat);
% Handle variations in struct/variable naming inside basin_map.mat
if isfield(basin_data, 'basin_map')
    basin_mask = basin_data.basin_map; % Expecting 720x360 or 360x720 matrix with IDs 1..103
elseif isfield(basin_data, 'basins')
    basin_mask = basin_data.basins;
else
    fn = fieldnames(basin_data);
    basin_mask = basin_data.(fn{1});
end

% Standardize orientation to [720 x 360] (Lon x Lat) if transposed
if size(basin_mask, 1) == 360 && size(basin_mask, 2) == 720
    basin_mask = basin_mask';
end

n_basins = max(basin_mask(:));
fprintf('Found %d major river basins in mask matrix.\n', n_basins);

%% Construct Latitude Cosine Weighting Matrix
% For 0.5-degree global grid (-89.75 to 89.75 lat, -179.75 to 179.75 lon)
lat_coords = -89.75:0.5:89.75; % 360 latitude points
lon_coords = -179.75:0.5:179.75; % 720 longitude points

% Compute latitude weight vector: w(lat) = cosd(lat)
cos_weights_1d = cosd(lat_coords); % 1 x 360 vector
cos_weights_2d = repmat(cos_weights_1d, 720, 1); % 720 x 360 spatial weight grid

%% 1. Aggregate GRACE TWS Anomalies
% -------------------------------------------------------------------------
fprintf('[1/6] Extracting basin time-series for TWS Anomalies...\n');
grids = load(grid_mat, 'tws_grid');
if ~isempty(grids.tws_grid)
    TWS_basin = extract_weighted_basin_series(grids.tws_grid, basin_mask, cos_weights_2d, n_basins);
else
    TWS_basin = [];
end
clear grids; % Memory Directive: Clear heavy 3D grid immediately

%% 2. Aggregate Precipitation (P)
% -------------------------------------------------------------------------
fprintf('[2/6] Extracting basin time-series for Precipitation (P)...\n');
grids = load(grid_mat, 'P_grid');
if ~isempty(grids.P_grid)
    P_basin = extract_weighted_basin_series(grids.P_grid, basin_mask, cos_weights_2d, n_basins);
else
    P_basin = [];
end
clear grids; % Memory Directive: Clear heavy 3D grid immediately

%% 3. Aggregate Evapotranspiration (ET)
% -------------------------------------------------------------------------
fprintf('[3/6] Extracting basin time-series for Evapotranspiration (ET)...\n');
grids = load(grid_mat, 'ET_grid');
if ~isempty(grids.ET_grid)
    ET_basin = extract_weighted_basin_series(grids.ET_grid, basin_mask, cos_weights_2d, n_basins);
else
    ET_basin = [];
end
clear grids; % Memory Directive: Clear heavy 3D grid immediately

%% 4. Aggregate Runoff / Discharge (Q)
% -------------------------------------------------------------------------
fprintf('[4/6] Extracting basin time-series for Runoff/Discharge (Q)...\n');
grids = load(grid_mat, 'Q_grid');
if ~isempty(grids.Q_grid)
    Q_basin = extract_weighted_basin_series(grids.Q_grid, basin_mask, cos_weights_2d, n_basins);
else
    Q_basin = [];
end
clear grids; % Memory Directive: Clear heavy 3D grid immediately

%% 5. Aggregate Groundwater Abstraction (GW_abs)
% -------------------------------------------------------------------------
fprintf('[5/6] Extracting basin time-series for GW Abstraction...\n');
grids = load(grid_mat, 'GW_grid');
if ~isempty(grids.GW_grid)
    GW_basin = extract_weighted_basin_series(grids.GW_grid, basin_mask, cos_weights_2d, n_basins);
else
    GW_basin = [];
end
clear grids; % Memory Directive: Clear heavy 3D grid immediately

%% 6. Aggregate Surface Water Abstraction (SW_abs)
% -------------------------------------------------------------------------
fprintf('[6/6] Extracting basin time-series for SW Abstraction...\n');
grids = load(grid_mat, 'SW_grid');
if ~isempty(grids.SW_grid)
    SW_basin = extract_weighted_basin_series(grids.SW_grid, basin_mask, cos_weights_2d, n_basins);
else
    SW_basin = [];
end
clear grids; % Memory Directive: Clear heavy 3D grid immediately

%% Load Date Vector
dates_file = fullfile(raw_dir, 'grace_dates.mat');
if exist(dates_file, 'file')
    date_struct = load(dates_file);
    fn = fieldnames(date_struct);
    grace_dates = date_struct.(fn{1});
else
    % Default monthly time sequence 2002 to 2021 (240 months)
    grace_dates = datetime(2002, 1, 1) + calmonths(0:239)';
end

%% Save Basin Time-Series File
out_mat = fullfile(processed_dir, 'basin_time_series.mat');
fprintf('Saving aggregated 2D basin time-series matrices to %s...\n', out_mat);
save(out_mat, 'P_basin', 'ET_basin', 'Q_basin', 'GW_basin', 'SW_basin', 'TWS_basin', 'grace_dates', '-v7.3');
fprintf('=== STEP 2 Complete: 103-Basin Cosine-Weighted Time-Series Saved ===\n\n');

%% Helper Function for Latitude Cosine-Weighted Spatial Aggregation
function basin_series = extract_weighted_basin_series(grid_3d, basin_mask, cos_weights, n_basins)
    % INPUTS:
    %   grid_3d     - 3D matrix [Lon x Lat x Time] (e.g. 720 x 360 x N_time)
    %   basin_mask  - 2D matrix [720 x 360] containing basin IDs (1..n_basins)
    %   cos_weights - 2D matrix [720 x 360] containing cosd(latitude)
    %   n_basins    - Total number of basins (103)
    % OUTPUT:
    %   basin_series - 2D matrix [N_time x n_basins]
    
    [n_lon, n_lat, n_time] = size(grid_3d);
    basin_series = nan(n_time, n_basins);
    
    % Loop over each basin ID
    for b = 1:n_basins
        basin_idx = (basin_mask == b);
        if ~any(basin_idx(:))
            continue;
        end
        
        % Extract spatial weights for this basin
        w_b = cos_weights(basin_idx);
        
        % Vectorized time extraction across all timesteps for this basin
        for t = 1:n_time
            slice_t = grid_3d(:, :, t);
            vals_t  = slice_t(basin_idx);
            
            % Filter out NaNs for weighted average computation
            valid_mask = ~isnan(vals_t);
            if any(valid_mask)
                basin_series(t, b) = sum(vals_t(valid_mask) .* w_b(valid_mask)) / sum(w_b(valid_mask));
            end
        end
    end
end
