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

fprintf('=== STEP 2: Starting Latitude Cosine-Weighted Basin Aggregation ===\n');

%% Directory Setup & Loading Inputs
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

processed_dir = fullfile(project_root, 'data', 'processed');
raw_dir       = fullfile(project_root, 'data', 'raw');
grace_dir     = fullfile(project_root, 'grace');

% Check if standardized grids are present in memory. If not, load or compute them.
if ~exist('tws_grid', 'var') && ~exist('P_grid', 'var')
    grid_mat = fullfile(processed_dir, 'standardized_grids.mat');
    if exist(grid_mat, 'file')
        fprintf('Loading standardized spatial grids from %s...\n', grid_mat);
        load(grid_mat);
    else
        fprintf('Standardized spatial grids not in memory. Running step01_unit_conversion.m...\n');
        run(fullfile(project_root, 'src', 'preprocessing', 'step01_unit_conversion.m'));
    end
end

basin_mat = fullfile(processed_dir, 'basin_map.mat');
if ~exist(basin_mat, 'file')
    basin_mat = fullfile(raw_dir, 'basin_map.mat');
end

if ~exist(basin_mat, 'file')
    error('Basin map file %s not found in data/raw/ or data/processed/!', basin_mat);
end

fprintf('Loading basin masks from %s...\n', basin_mat);
basin_data = load(basin_mat);
% Handle basin_map variable inside basin_map.mat (values 1..103, non-basins NaN)
if isfield(basin_data, 'basin_map')
    basin_mask = double(basin_data.basin_map);
elseif isfield(basin_data, 'basins')
    basin_mask = double(basin_data.basins);
else
    fn = fieldnames(basin_data);
    basin_mask = double(basin_data.(fn{1}));
end

% Get total number of basins ignoring NaNs
n_basins = max(basin_mask(~isnan(basin_mask)));
if isempty(n_basins)
    n_basins = 103;
else
    n_basins = double(n_basins);
end
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
if exist('tws_grid', 'var') && ~isempty(tws_grid)
    TWS_basin = extract_weighted_basin_series(tws_grid, basin_mask, cos_weights_2d, n_basins);
    clear tws_grid; % Memory Directive: Clear heavy 3D grid immediately
else
    TWS_basin = [];
end

%% 2. Aggregate Precipitation (P)
% -------------------------------------------------------------------------
fprintf('[2/6] Extracting basin time-series for Precipitation (P)...\n');
if exist('P_grid', 'var') && ~isempty(P_grid)
    P_basin = extract_weighted_basin_series(P_grid, basin_mask, cos_weights_2d, n_basins);
    clear P_grid; % Memory Directive: Clear heavy 3D grid immediately
else
    P_basin = [];
end

%% 3. Aggregate Evapotranspiration (ET)
% -------------------------------------------------------------------------
fprintf('[3/6] Extracting basin time-series for Evapotranspiration (ET)...\n');
if exist('ET_grid', 'var') && ~isempty(ET_grid)
    ET_basin = extract_weighted_basin_series(ET_grid, basin_mask, cos_weights_2d, n_basins);
    clear ET_grid; % Memory Directive: Clear heavy 3D grid immediately
else
    ET_basin = [];
end

%% 4. Aggregate Runoff / Discharge (Q)
% -------------------------------------------------------------------------
fprintf('[4/6] Extracting basin time-series for Runoff/Discharge (Q)...\n');
if exist('Q_grid', 'var') && ~isempty(Q_grid)
    Q_basin = extract_weighted_basin_series(Q_grid, basin_mask, cos_weights_2d, n_basins);
    clear Q_grid; % Memory Directive: Clear heavy 3D grid immediately
else
    Q_basin = [];
end

%% 5. Aggregate Groundwater Abstraction (GW_abs)
% -------------------------------------------------------------------------
fprintf('[5/6] Extracting basin time-series for GW Abstraction...\n');
if exist('GW_grid', 'var') && ~isempty(GW_grid)
    GW_basin = extract_weighted_basin_series(GW_grid, basin_mask, cos_weights_2d, n_basins);
    clear GW_grid; % Memory Directive: Clear heavy 3D grid immediately
else
    GW_basin = [];
end

%% 6. Aggregate Surface Water Abstraction (SW_abs)
% -------------------------------------------------------------------------
fprintf('[6/6] Extracting basin time-series for SW Abstraction...\n');
if exist('SW_grid', 'var') && ~isempty(SW_grid)
    SW_basin = extract_weighted_basin_series(SW_grid, basin_mask, cos_weights_2d, n_basins);
    clear SW_grid; % Memory Directive: Clear heavy 3D grid immediately
else
    SW_basin = [];
end

%% Construct Continuous Monthly Date Vector (Apr 2002 to Dec 2019 = 213 months)
grace_dates = (datetime(2002, 4, 1) + calmonths(0:212))';

%% Save Basin Time-Series 2D Matrices to Disk
ts_file = fullfile(processed_dir, 'basin_time_series.mat');
fprintf('Saving basin time-series to %s...\n', ts_file);
save(ts_file, 'P_basin', 'ET_basin', 'Q_basin', 'GW_basin', 'SW_basin', 'TWS_basin', 'grace_dates', '-v7.3');
fprintf('=== STEP 2 Complete: 103-Basin Time-Series Aggregated & Saved ===\n\n');

%% Helper Function for Latitude Cosine-Weighted Spatial Aggregation
function basin_series = extract_weighted_basin_series(grid_3d, basin_mask, cos_weights, n_basins)
    % INPUTS:
    %   grid_3d     - 3D matrix [Lon x Lat x Time] or [Lat x Lon x Time]
    %   basin_mask  - 2D matrix matching spatial dimensions of grid_3d (1..103, NaNs for non-basins)
    %   cos_weights - 2D matrix matching spatial dimensions of grid_3d
    %   n_basins    - Total number of basins (103)
    % OUTPUT:
    %   basin_series - 2D matrix [N_time x n_basins]
    
    [n_dim1, n_dim2, n_time] = size(grid_3d);
    
    % Ensure basin_mask and cos_weights match spatial dimensions of grid_3d
    if size(basin_mask, 1) ~= n_dim1 || size(basin_mask, 2) ~= n_dim2
        if size(basin_mask, 1) == n_dim2 && size(basin_mask, 2) == n_dim1
            basin_mask = basin_mask';
        end
    end
    if size(cos_weights, 1) ~= n_dim1 || size(cos_weights, 2) ~= n_dim2
        if size(cos_weights, 1) == n_dim2 && size(cos_weights, 2) == n_dim1
            cos_weights = cos_weights';
        end
    end
    
    basin_series = nan(n_time, n_basins);
    
    % Loop over each basin ID from 1 to n_basins
    for b = 1:n_basins
        basin_idx = (basin_mask == b);
        if ~any(basin_idx(:))
            continue;
        end
        
        % Extract spatial weights for this basin
        w_b = cos_weights(basin_idx);
        
        % Compute spatial weighted average across all timesteps for this basin
        for t = 1:n_time
            slice_t = grid_3d(:, :, t);
            vals_t  = slice_t(basin_idx);
            
            % Filter out NaNs in data and weights
            valid_mask = ~isnan(vals_t) & ~isnan(w_b);
            if any(valid_mask)
                basin_series(t, b) = sum(vals_t(valid_mask) .* w_b(valid_mask)) / sum(w_b(valid_mask));
            end
        end
    end
end
