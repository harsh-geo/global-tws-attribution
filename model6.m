%% Introduction
% Why Model 6? 
%% Data loading
precip = ncread("precip_era5_monthly_2002_2019.nc",'tp');
precip = precip * 100; % converted to cm/month
disp('Loaded precipitation...');
et = ncread("et_gleam_global_2002_2021.nc",'E');
et(et == -999) = NaN; % Updated fill value
et = et / 10;
disp('Loaded evapotranspiration...');
gw_abs = ncread("gw_abs_pcrglobwb_2002_2019.nc",'total_groundwater_abstraction');
gw_abs(gw_abs > 1e19) = NaN;
gw_abs = gw_abs * 100;
disp('Loaded groundwater abstraction...');
sw_abs = ncread("sw_abs_pcrglobwb_2002_2019.nc",'surface_water_abstraction');
sw_abs(sw_abs > 1e19) = NaN;
sw_abs = sw_abs * 100;
disp('Loaded surface water abstraction...');

tws = ncread("grace\grace_continuous2.nc",'lwe_thickness');
disp('Loaded GRACE TWS');
%% Computing TWSC
% tws is lon x lat x time
sz = size(tws);
ntime = sz(3);

% Preallocate twsc and set edges to NaN
twsc = nan(sz);

% Compute centered difference for t = 2..ntime-1
% Use vectorized indexing
twsc(:,:,2:ntime-1) = (tws(:,:,3:ntime) - tws(:,:,1:ntime-2)) / 2;

% Ensure first and last time slices remain NaN (already set)
%% Spatial aggregation
% I will use a cosine-weighted basin spatial mean
time_series
%% RF Modeling
% Use block CV on TWSC target

%% Validation
% Will perform residual and other checks here
