% This script basically computes the spatial mean of a dataset
% Resolution of input dataset should be 0.5 degrees spatially. 
% Ensure basin_map.mat file is already present in the same directory.
%% Preparing spatial weights
load('basin_map.mat','basin_map');
[nLon, nLat] = size(basin_map);
lat_vec = ncread('et_gleam_global_2002_2021.nc','lat');
lon_vec = ncread('et_gleam_global_2002_2021.nc','lon');

weights_1d = cosd(lat_vec(:))';
weights_2d = repmat(weights_1d, nLon, 1);
basin_map_flat = basin_map(:);
weights_flat = weights_2d(:);
num_basins = max(basin_map_flat);

basin_names = arrayfun(@(x) sprintf('Basin_%d', x), 1:num_basins, ...
    'UniformOutput',false);
basin_coords = zeros(num_basins, 2);
[Lon_grid, Lat_grid] = ndgrid(lon_vec, lat_vec);
for b = 1:num_basins
    idx = (basin_map == b);
    if any(idx(:))
        basin_coords(b,1) = mean(Lat_grid(idx));
        basin_coords(b,2) = mean(Lon_grid(idx));
    end
end
%% 
function basin_ts = aggregate_basin_data(data_cube, basin_map_flat, ...
    weights_flat, num_basins)
[nLon, nLat, nTime] = size(data_cube); data_flat = reshape(data_cube, ...
    [nLon * nLat, nTime]); basin_ts = zeros(nTime, num_basins);
for b = 1:num_basins
    in_basin = (basin_map_flat == b); if ~any( ...
            in_basin), basin_ts(:,b) = NaN; 
        continue; end
    basin_pixels = data_flat(in_basin,:); basin_weights = weights_flat( ...
        in_basin); valid_mask = ~isnan(basin_pixels);
    weight_matrix = repmat(basin_weights, 1, nTime); weight_matrix( ...
        ~valid_mask) = 0; basin_pixels(~valid_mask) = 0;
    basin_ts(:,b) = sum(basin_pixels .* weight_matrix, 1) ./ sum( ...
        weight_matrix, 1);
end
end
%% Aggregating basin data into time series
basin_precip_ts = aggregate_basin_data(precip, basin_map_flat, ...
    weights_flat, num_basins);
basin_et_ts = aggregate_basin_data(et, basin_map_flat, ...
    weights_flat, num_basins);
basin_gw_abs_ts = aggregate_basin_data(gw_abs, basin_map_flat, ...
    weights_flat, num_basins);
basin_sw_abs_ts = aggregate_basin_data(sw_abs, basin_map_flat, ...
    weights_flat, num_basins);
disp('Computed basinal data into time series...');
basin_tws_ts = aggregate_basin_data(tws, basin_map_flat, ...
    weights_flat, num_basins);
% Ensure that after running this code all the unnecessary variables are
% cleared so as to reduce space