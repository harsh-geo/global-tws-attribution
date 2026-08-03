# 1. Fix the Grid Type (Generic -> LonLat)
cdo setgrid,../precip_DJF.nc grace_corrected_monthly.nc grace_fixed_grid.nc

# 2. Fill Time Gaps (Linear Interpolation)
# This forces the file to have a continuous monthly axis from start to end.
# Gaps (missing GRACE months) will be filled linearly.
cdo inttime,2002-04-01,00:00:00,1mo grace_fixed_grid.nc grace_continuous.nc

# 3. Create the 4 Seasons (Aligning with Precip/Temp)
cdo seasmean -selseas,DJF grace_continuous.nc tws_DJF.nc
cdo seasmean -selseas,MAM grace_continuous.nc tws_MAM.nc
cdo seasmean -selseas,JJA grace_continuous.nc tws_JJA.nc
cdo seasmean -selseas,SON grace_continuous.nc tws_SON.nc
