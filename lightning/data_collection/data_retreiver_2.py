import xarray as xr
import cdsapi
import netCDF4 as nc
from goes2go import GOES
import requests
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from numba import njit, prange

# To retrieve the ERA5 data from the climate data store, use the cdsapi library.
# It requires you to make an account and to have an API key, which I have already done.

# There are two ERA5 datasets which we are interested in, reanalysis on single levels
# and reanalysis on pressure levels. The primary difference is that the data on pressure levels
# has an extra variable which refers to the pressure levels at which the data is obtained.


# To remove the need to memorize which variables are on single levels or not,
# this function will take in the variable nickname as input and verify the level type on its own.

import cdsapi

# Variables for single-level and pressure-level datasets
single_lvl_variables = ["total_precipitation", "convective_available_potential_energy"]
pressure_lvl_variables = ["fraction_of_cloud_cover", "specific_cloud_ice_water_content", 
                          "specific_humidity", "temperature", "vertical_velocity"]

# Generate a list of times for a day's worth of data
time_list = [f"{str(hour).zfill(2)}:00" for hour in range(24)]  # Corrected to 24 hours

# Boundaries [N, W, S, E]


# Retrieving GOES GLM-LCFA or ABI-XXXX Data using goes2py
# There are two options, either download the data directly
# or produce a pandas dataframe of the data for a specific range of time
# Note that the data is recorded minutely.


#######################################################################################################################

# Retrieving the GOES-GLMF (Flash Extent Density variable)

# Function for downloading code using a HTTP GET request. If the file does not exist,
# it will return None, and skip that file.
@njit(parallel=True)
def compute_latlon(x, y):
    """Computes lat/lon from GOES-GLM grid using numba for speed."""
    
    lat = np.zeros_like(x)
    lon = np.zeros_like(x)
    r_eq =  6378137.0
    r_pol = 6356752.31414
    l_0 = -75 * (np.pi/180)
    h_sat = 35786 * 10**3
    H = r_eq + h_sat


    for i in prange(x.shape[0]):  # Parallelize over rows
        for j in range(x.shape[1]):  # Iterate over columns
            x_val = x[i, j]
            y_val = y[i, j]

            # Avoid unnecessary function calls
            sin_x2 = np.sin(x_val)**2
            cos_x2 = np.cos(x_val)**2
            cos_y2 = np.cos(y_val)**2
            sin_y2 = np.sin(y_val)**2 * (r_eq**2 / r_pol**2)
            
            a = sin_x2 + cos_x2 * (cos_y2 + sin_y2)
            b = -2 * H * np.cos(x_val) * np.cos(y_val)
            c = H**2 - r_eq**2
            
            r_s = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            
            # Compute Cartesian coordinates
            s_x = r_s * np.cos(x_val) * np.cos(y_val)
            s_y = -r_s * np.sin(x_val)
            s_z = r_s * np.cos(x_val) * np.sin(y_val)
            
            # Compute latitude and longitude
            lat[i, j] = np.rad2deg(np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H - s_x)**2 + s_y**2))))
            lon[i, j] = np.rad2deg(l_0 - np.arctan2(s_y, H - s_x))
    
    return lat, lon

def calc_latlon(ds):
    # The math for this function was taken from 
    # https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm
    x = ds.x
    y = ds.y
    
    x,y = np.meshgrid(x,y)
    
    
    lat, lon = compute_latlon(x,y)
    
    ds = ds.assign_coords({
        "lat":(["y","x"],lat),
        "lon":(["y","x"],lon)
    })
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    
    return ds

def get_xy_from_latlon(ds, lats, lons):
    """ This function takes as input a desired set of lat and lon boundaries
        and returns the corresponding x and y coordinate boundaries.
        This allows users to bound the dataset, which is especially useful
        due to the fact that the data has nan lat/lon coordinates at the boundary.

        >>> lats = (25, 45) #S, N
        >>> lons = (-110, -70) # W, E
        >>> ((x1,x2), (y1, y2)) = get_xy_from_latlon(ds, lats, lons)
        >>> subset = ds.sel(x=slice(x1, x2), y=slice(y2, y1)) #Dataset in lat/lon box
    """
    lat1, lat2 = lats
    lon1, lon2 = lons

    lat = ds.lat.data
    lon = ds.lon.data
    
    x = ds.x.data
    y = ds.y.data
    
    x,y = np.meshgrid(x,y)
    
    x = x[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]
    y = y[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)] 
    
    return ((min(x), max(x)), (min(y), max(y)))

# The lat/lon coordinates are a 2D array because they depend on both y and x
# Important Note: Lat/Lon data is unusable unless you take a subset in an area further away from the boundaries due to NaNs

# New dataset with x and y coordinates removed, to make interpolation possible
def regrid_data(goes_ds, target_ds):
    """ Takes as input a GOES satellite dataset (ABI or GLM) and returns a regridded, coarsened dataset
        with coordinates of latitude and longitude. This function disregards data quality flags, as they lose
        meaning after interpoaltion. This function assumes that the desired lat and lon indices have already been
        sliced from the fed data.

        >>> fed_ds = xr.open_dataset(file_path)
        >>> era5_ds = xr.open_dataset(file_path)
        >>> regridded_fed = regrid_data(fed_ds, era5_ds) 
    """
    new_ds = xr.Dataset(
        {
            "Flash_extent_density": (("y", "x"), goes_ds.data)  # Original FED data
        },
        coords={
            "lat": (("y", "x"), goes_ds.lat.data),  # 2D latitude array
            "lon": (("y", "x"), goes_ds.lon.data)   # 2D longitude array
        }
    )

    # Now I'll interpolate it to the era5 grid. Note that the data itself has not changed yet, only the coordinates have. This transformation is accurate as it is the same
    # as the NOAA provided coordinate transformation found at: https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php 

    from scipy.interpolate import griddata

    # Flatten the 2D lat, lon, and data arrays from the original dataset because scipy interpolate does not work unless the data is in 1D
    lat_flat = new_ds.lat.values.ravel() 
    lon_flat = new_ds.lon.values.ravel()
    data_flat = new_ds.Flash_extent_density.values.ravel()

    # Create a meshgrid of the target lat/lon on the ERA5 grid
    target_lon, target_lat = np.meshgrid(target_ds.longitude.values, target_ds.latitude.values)

    # Interpolate using scipy's griddata
    interpolated_data = griddata(
        points=(lon_flat, lat_flat),
        values=data_flat,
        xi=(target_lon, target_lat),
        method='linear'
    )

    # Wrap the result back into an xarray DataArray with ERA5 coordinates
    interpolated_da = xr.DataArray(
        interpolated_data,
        coords={
            "latitude": target_ds.latitude,  # Match coordinate with era5 to ensure interpolation worked, if it didnt an error would occur here
            "longitude": target_ds.longitude
        },
        dims=["latitude", "longitude"]
    )

    return interpolated_da

def interp_to_grid(fed_hour, bounds, target):
    fed_hour = calc_latlon(fed_hour)
    ((x1,x2), (y1, y2)) = get_xy_from_latlon(fed_hour, bounds[0], bounds[1])
    fed_hour = fed_hour.sel(x=slice(x1, x2), y=slice(y2, y1))
    fed_hour = regrid_data(fed_hour, target)
    fed_hour = fed_hour.fillna(0)
    return None


def download_file(url, local_filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        print(f"Error downloading file {url}: {e}")
        return None

# Using the FED Window we can go from downloading 60 files to only 12 files, and since that is a low number of files,
# we can process all of their data, and once we are done processing the data for each hour, we can delete the files.
def get_fed_data_for_hour(year, day_of_year, hour):
    # URL provided from Henri's Code
    base_url = "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/"
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour)
    year_str = date.strftime('%Y')
    day_str = date.strftime('%j')
    url = f"{base_url}{year_str}/{year_str}{day_str.zfill(3)}/"

    files = []

    # Start looping through the current day files
    for minute in range(5, 65, 5):
        time_str = (date + timedelta(minutes=minute)).strftime('%Y%m%d%H%M%S')
        file_pattern = f"OR_GLM-L2-GLMF-M3_G16_e{time_str}.nc"
        files.append(url + file_pattern)
    
    return files

# This function uses the URL of the file from the website to get the name of the file as the
# final index of the link string is the GLM file name. We run try and except to handle any missing file errors.
# We then acquire the FED over the 5-minute window in each file which will later be summed for the hour.
def process_file(file_url):
    local_file = file_url.split('/')[-1]
    downloaded_file = download_file(file_url, local_file)
    if downloaded_file is None:
        return None, file_url

    try:
        ds = xr.open_dataset(downloaded_file)
        fed_window = ds['Flash_extent_density_window']
        ds.close()
        return fed_window, None
    except Exception as e:
        print(f"Error processing file {file_url}: {e}")
        return None, file_url

# This method sums the FED window for each of the 12 files within an hour.
def aggregate_fed_hour(year, day_of_year, hour):
    files = get_fed_data_for_hour(year, day_of_year, hour)
    hourly_sum = None
    faulty_links = [] # Tracking links that do not exist

    print(f"Processing files for year {year}, day {day_of_year}, hour {hour}")
    print(f"Files to process: {files}") # Tracking the file names to ensure that the right files are being processed

    # Using parallel computing library to carry out aggregation operation simultaneously.
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_file, file_url): file_url for file_url in files}

        for future in as_completed(futures):
            fed_window, faulty_link = future.result()
            if fed_window is not None:
                if hourly_sum is None:
                    hourly_sum = fed_window
                else:
                    hourly_sum += fed_window
            if faulty_link is not None:
                faulty_links.append(faulty_link)

    # Delete files after processing
    for file_url in files:
        local_file = file_url.split('/')[-1]
        if os.path.exists(local_file):
            os.remove(local_file)
    # Regrid the data on a specifc lat/lon grid
    if hourly_sum is not None:
        hourly_sum = hourly_sum.expand_dims('time')
        hourly_sum['time'] = [datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour+1)]
        
    return hourly_sum, faulty_links


# Main function to execute the workflow for a specific day
def retrieve_goes_glmf(date: datetime, output_file):
    all_hours_data = []
    all_faulty_links = []
    year = date.year
    day_of_year = date.timetuple().tm_yday

    for hour in range(24):
        hourly_fed, faulty_links = aggregate_fed_hour(year, day_of_year, hour)
        if hourly_fed is not None:
            all_hours_data.append(hourly_fed)
        all_faulty_links.extend(faulty_links)
    
    if all_hours_data:
        combined_fed = xr.concat(all_hours_data, dim='time')
        combined_fed.name = 'FED_Window'
        combined_fed.to_net_cdf(output_file)
        print(f"Saved 24-hour FED data to {output_file}")
    else:
        print("No valid data to save.")
    
    with open("faulty_links.txt", "w") as f:
        for link in all_faulty_links:
            f.write(f"{link}\n")

    print(f"Faulty links have been saved to faulty_links.txt")

def retrieve_goes_abi(var_name, date_string, bounds, target):
    variable_dictionary = {'LCFA': 'GLM-L2-LCFA', 'HT': 'ABI-L2-ACHAF', 'TEMP':'ABI-L2-ACHTF','Clear Sky Mask': 'ABI-L2-ACMF',
                        'Cloud Optical Depth': 'ABI-L2-CODF', 'CAPE': 'ABI-L2-DSIF',
                        'Land Surface Temp': 'ABI-L2-LSTF', 'RRQPE': 'ABI-L2-RRQPEF'} # the keys will be the variable nicknames, and the values will be the product names

    G = GOES(satellite=16, product=variable_dictionary[var_name], domain='F')
    ds = xr.concat([G.nearesttime(f"{date_string} {str(hour).zfill(2)}:00") for hour in range(0, 24)], dim='t')    
    da_list = []
    ds_all = ds[var_name].fillna(0)
    for time in ds_all.t.values:
        ds_time = ds_all.sel(t=time)  # Select data for the current time
        ds_time = calc_latlon(ds_time)  # Apply latitude-longitude calculation
        # Get bounding box coordinates for the region of interest
        ((x1, x2), (y1, y2)) = get_xy_from_latlon(ds_time, bounds[0], bounds[1])
        # Subset the data based on the bounding box
        ds_time = ds_time.sel(x=slice(x1, x2), y=slice(y2, y1))
        # Regrid the data to match the target dataset
        ds_time = regrid_data(ds_time, target)
        # Append the processed data for the current time to the list
        da_list.append(ds_time)
    da = xr.concat(da_list, dim='t')
    da['t'] = target.time.values
    da = da.rename({'t': 'time'})
    return da


