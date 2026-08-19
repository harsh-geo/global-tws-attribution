import cdsapi
import os

def download_era5_temperature():
    c = cdsapi.Client()

    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    output_file = os.path.join(raw_dir, 'era5_t2m_2002_2019.nc')
    
    print(f"Queueing ERA5 2m Temperature download to {output_file}...")
    
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': '2m_temperature',
            'year': [str(y) for y in range(2002, 2020)],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'time': '00:00',
        },
        output_file)
        
    print("Download completed.")

if __name__ == '__main__':
    download_era5_temperature()
