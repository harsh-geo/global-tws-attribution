import scipy.io as sio
import numpy as np
import pandas as pd
import os
import warnings

# Suppress runtime warnings for NaNs
warnings.filterwarnings('ignore')

print("Loading basin map...")
mat = sio.loadmat('data/processed/basin_map.mat')
basin_mask = mat['basin_map']

# dimensions are 720 (lon) x 360 (lat)
lat = np.linspace(-89.75, 89.75, 360)
lon = np.linspace(-179.75, 179.75, 720)

print("Calculating grid cell areas...")
R = 6371.0 # km
deg2rad = np.pi / 180.0
dlon = 0.5 * deg2rad

# Create 2D arrays
# basin_mask is [lon, lat], so we need Lat to be broadcastable to [720, 360]
# Lat should vary along the second axis
Lat = lat[np.newaxis, :] # shape (1, 360)
# Area calculation depends only on latitude
Area_grid = (R**2) * dlon * np.abs(np.sin((Lat + 0.25)*deg2rad) - np.sin((Lat - 0.25)*deg2rad))
Area_grid = np.broadcast_to(Area_grid, (720, 360))

print("Aggregating areas by basin...")
num_basins = int(np.nanmax(basin_mask))
basin_areas = {}
for b in range(1, num_basins + 1):
    idx = (basin_mask == b)
    basin_areas[b] = np.nansum(Area_grid[idx])

# Read and update Table_S1_Supplementary.csv
s1_path = 'outputs/tables/Table_S1_Supplementary.csv'
if os.path.exists(s1_path):
    print("Updating Table_S1_Supplementary.csv...")
    df = pd.read_csv(s1_path)
    
    df['Area_km2'] = df['Basin_ID'].map(basin_areas)
    # Trend is in cm/yr. Area is in km2. 1 cm = 1e-5 km.
    df['Trend_km3_yr'] = df['Trend_cm_yr'] * df['Area_km2'] * 1e-5
    
    df.to_csv(s1_path, index=False)
    print("Table_S1_Supplementary.csv updated successfully.")
    
    stressed_basins = [42, 51, 39, 40, 45, 14, 5, 17, 36]
    for b in stressed_basins:
        row = df[df['Basin_ID'] == b]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"Basin {b} ({row['Basin_Name']}): Area = {row['Area_km2']:.0f} km2, Trend = {row['Trend_km3_yr']:.3f} km3/yr")

summary_path = 'outputs/tables/basin_summary_table.csv'
if os.path.exists(summary_path):
    print("Updating basin_summary_table.csv...")
    df_sum = pd.read_csv(summary_path)
    df_sum['Area_km2'] = df_sum['Basin_ID'].map(basin_areas)
    df_sum['Trend_km3_yr'] = df_sum['Trend_cm_yr'] * df_sum['Area_km2'] * 1e-5
    if os.path.exists(s1_path):
        name_map = df.set_index('Basin_ID')['Basin_Name'].to_dict()
        df_sum['Basin_Name'] = df_sum['Basin_ID'].map(name_map)
    df_sum.to_csv(summary_path, index=False)
    print("basin_summary_table.csv updated successfully.")
