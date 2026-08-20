%% step02b_extract_gridwise.m - Extract Time-Series for Every Pixel in 103 Basins
% =========================================================================
% AUTHOR: Computational Hydrology & HPC Pipeline
% PROJECT: Global Terrestrial Water Storage (TWS) Attribution
% =========================================================================
% PURPOSE:
%   1. Load standardized 3D spatio-temporal grids (720x360xN_time).
%   2. Load 103 major global river basin masks.
%   3. Instead of spatial aggregation, extract the time series for EVERY 
%      individual valid pixel within each basin.
%
% OUTPUT:
%   - data/processed/gridwise_time_series.mat containing cell arrays (103x1):
%     P_gridwise, ET_gridwise, Q_gridwise, GW_gridwise, SW_gridwise, TWS_gridwise
%     pixel_coords (Lat, Lon indices for each pixel)
% =========================================================================

fprintf('=== STEP 2b: Starting Grid-wise Pixel Extraction ===\n');

%% Directory Setup & Loading Inputs
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
if isempty(project_root) || ~exist(fullfile(project_root, 'data'), 'dir')
    project_root = pwd;
end

processed_dir = fullfile(project_root, 'data', 'processed');
raw_dir       = fullfile(project_root, 'data', 'raw');

% Check if standardized grids are present in memory
if ~exist('tws_grid', 'var') || ~exist('P_grid', 'var')
    grid_mat = fullfile(processed_dir, 'standardized_grids.mat');
    if exist(grid_mat, 'file')
        fprintf('Loading standardized spatial grids from %s...\n', grid_mat);
        load(grid_mat);
    else
        fprintf('Standardized spatial grids not found! Run step01 first.\n');
        return;
    end
end

basin_mat = fullfile(processed_dir, 'basin_map.mat');
if ~exist(basin_mat, 'file')
    basin_mat = fullfile(raw_dir, 'basin_map.mat');
end

fprintf('Loading basin masks from %s...\n', basin_mat);
basin_data = load(basin_mat);
if isfield(basin_data, 'basin_map')
    basin_mask = double(basin_data.basin_map);
elseif isfield(basin_data, 'basins')
    basin_mask = double(basin_data.basins);
else
    fn = fieldnames(basin_data);
    basin_mask = double(basin_data.(fn{1}));
end

% Ensure basin_mask is same orientation as grids
[n_dim1, n_dim2, n_time] = size(P_grid);
if size(basin_mask, 1) ~= n_dim1 || size(basin_mask, 2) ~= n_dim2
    if size(basin_mask, 1) == n_dim2 && size(basin_mask, 2) == n_dim1
        basin_mask = basin_mask';
    end
end

n_basins = max(basin_mask(~isnan(basin_mask)));
if isempty(n_basins), n_basins = 103; else, n_basins = double(n_basins); end
fprintf('Found %d major river basins.\n', n_basins);

%% Initialize Cell Arrays
TWS_gridwise = cell(n_basins, 1);
P_gridwise   = cell(n_basins, 1);
ET_gridwise  = cell(n_basins, 1);
Q_gridwise   = cell(n_basins, 1);
GW_gridwise  = cell(n_basins, 1);
SW_gridwise  = cell(n_basins, 1);
pixel_coords = cell(n_basins, 1); % Store (dim1, dim2) for each pixel

%% Extraction Loop
fprintf('Extracting pixel time-series for %d basins...\n', n_basins);

% Set up waitbar
if usejava('desktop')
    h_wb = waitbar(0, 'Extracting grid-wise data...');
else
    h_wb = [];
end

for b = 1:n_basins
    if ~isempty(h_wb)
        waitbar(b / n_basins, h_wb, sprintf('Extracting basin %d / %d', b, n_basins));
    elseif mod(b, 10) == 0 || b == 1 || b == n_basins
        fprintf('  -> Extracting basin %d / %d...\n', b, n_basins);
    end
    % Find linear indices of all pixels belonging to basin b
    idx_b = find(basin_mask == b);
    n_pixels = length(idx_b);
    
    if n_pixels == 0
        continue;
    end
    
    % Store coordinates (dim1=lon/lat, dim2=lat/lon)
    [i_idx, j_idx] = ind2sub([n_dim1, n_dim2], idx_b);
    pixel_coords{b} = [i_idx, j_idx];
    
    % Initialize matrices for this basin: [n_time x n_pixels]
    tws_b = nan(n_time, n_pixels);
    p_b   = nan(n_time, n_pixels);
    et_b  = nan(n_time, n_pixels);
    q_b   = nan(n_time, n_pixels);
    gw_b  = nan(n_time, n_pixels);
    sw_b  = nan(n_time, n_pixels);
    
    for t = 1:n_time
        if exist('tws_grid', 'var') && ~isempty(tws_grid)
            slice_t = tws_grid(:,:,t);
            tws_b(t, :) = slice_t(idx_b);
        end
        if exist('P_grid', 'var') && ~isempty(P_grid)
            slice_t = P_grid(:,:,t);
            p_b(t, :) = slice_t(idx_b);
        end
        if exist('ET_grid', 'var') && ~isempty(ET_grid)
            slice_t = ET_grid(:,:,t);
            et_b(t, :) = slice_t(idx_b);
        end
        if exist('Q_grid', 'var') && ~isempty(Q_grid)
            slice_t = Q_grid(:,:,t);
            q_b(t, :) = slice_t(idx_b);
        end
        if exist('GW_grid', 'var') && ~isempty(GW_grid)
            slice_t = GW_grid(:,:,t);
            gw_b(t, :) = slice_t(idx_b);
        end
        if exist('SW_grid', 'var') && ~isempty(SW_grid)
            slice_t = SW_grid(:,:,t);
            sw_b(t, :) = slice_t(idx_b);
        end
    end
    
    TWS_gridwise{b} = tws_b;
    P_gridwise{b}   = p_b;
    ET_gridwise{b}  = et_b;
    Q_gridwise{b}   = q_b;
    GW_gridwise{b}  = gw_b;
    SW_gridwise{b}  = sw_b;
end

if exist('h_wb', 'var') && ~isempty(h_wb) && isvalid(h_wb)
    close(h_wb);
end

% Clear heavy 3D grids
clear tws_grid P_grid ET_grid Q_grid GW_grid SW_grid;

%% Dates
grace_dates = (datetime(2002, 4, 1) + calmonths(0:n_time-1))';

%% Save to Disk
out_file = fullfile(processed_dir, 'gridwise_time_series.mat');
fprintf('Saving grid-wise time-series to %s...\n', out_file);
save(out_file, 'TWS_gridwise', 'P_gridwise', 'ET_gridwise', 'Q_gridwise', ...
    'GW_gridwise', 'SW_gridwise', 'pixel_coords', 'grace_dates', '-v7.3');
fprintf('=== STEP 2b Complete ===\n\n');
