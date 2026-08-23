try
    disp('Loading basin_map...');
    load('data/processed/basin_map.mat');
    if ~exist('lat', 'var')
        disp('Loading tws_basins...');
        load('data/processed/tws_basins.mat');
    end

    R = 6371; % km
    deg2rad = pi/180;
    dlon = 0.5 * deg2rad;

    [Lon, Lat] = meshgrid(lon, lat);
    Area_grid = (R^2) * dlon * abs(sin((Lat + 0.25)*deg2rad) - sin((Lat - 0.25)*deg2rad));

    num_basins = 103;
    basin_areas = zeros(num_basins, 1);
    for b = 1:num_basins
        idx = (basin_mask == b);
        basin_areas(b) = sum(Area_grid(idx));
    end
    
    disp('Saving areas to basin_areas.csv...');
    csvwrite('outputs/tables/basin_areas.csv', basin_areas);
    disp('Success');
catch ME
    disp('Error occurred:');
    disp(ME.message);
    disp(ME.stack(1));
end
exit;
