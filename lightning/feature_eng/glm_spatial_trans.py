# Author: Maforikan Amoussou
# Date: 10/4/2024
# Description: Transforming satellite projection coordinates of GLM data
# into latitude and longitude, and interpolating the GLM/ABI data onto
# the ERA5 grid

import xarray as xr
import requests
import numpy as np
import netCDF4 as nc
import os
#import xesmf as xe
import cdsapi
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 


url = "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/2023/2023001/OR_GLM-L2-GLMF-M3_G16_e20230101000000.nc"
output_path = "/home/o/oneill/mamous3/lightning/datasets"
download_path = output_path + url[-41:]
os.makedirs(output_path, exist_ok=True)


response = requests.get(url)
with open(download_path, "wb") as file:
    file.write(response.content)
    print("Download complete.")

temp_fed = xr.open_dataset(download_path)
os.remove(download_path)

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "variable": ["convective_available_potential_energy"],
    "area": [45, -110, 25, -70]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download(output_path + "/era5_cape_temp.nc")
            
era5_ds = xr.open_dataset(output_path + "/era5_cape_temp.nc")
os.remove(output_path + "/era5_cape_temp.nc")

perspective_point_height = temp_fed.goes_imager_projection.attrs['perspective_point_height']  # in meters
semi_major_axis = temp_fed.goes_imager_projection.attrs['semi_major_axis']  # in meters
semi_minor_axis = temp_fed.goes_imager_projection.attrs['semi_minor_axis']  # in meters
longitude_of_projection_origin = temp_fed.goes_imager_projection.attrs['longitude_of_projection_origin']  # in degrees
x_scale_factor = 5.6e-05
x_add_offset = -0.151872
y_scale_factor = -5.6e-05
y_add_offset = 0.151872

def calc_latlon(ds):
    # The math for this function was taken from 
    # https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm
    x = ds.x
    y = ds.y
    
    x,y = np.meshgrid(x,y)
    
    r_eq = semi_major_axis
    r_pol = semi_minor_axis
    l_0 = longitude_of_projection_origin * (np.pi/180)
    h_sat = perspective_point_height
    H = r_eq + h_sat
    
    a = np.sin(x)**2 + (np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2))
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2
    
    r_s = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    
    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)
    
    lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H-s_x)**2 +s_y**2))) * (180/np.pi)
    lon = (l_0 - np.arctan(s_y / (H-s_x))) * (180/np.pi)
    
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

        >>> lats = (25, 45)
        >>> lons = (-110, -70)
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

temp_fed = calc_latlon(temp_fed)
lats = (25, 45) # S, N
lons = (-110, -70) # W, E
((x1,x2), (y1, y2)) = get_xy_from_latlon(temp_fed, lats, lons)
temp_fed_lat_lon = temp_fed.sel(x=slice(x1, x2), y=slice(y2, y1)) #Dataset in lat/lon box

# New dataset with x and y coordinates removed, to make interpolation possible, this is only possible if lat and lon are of the same dimensions as the data
fed_lat_lon = xr.Dataset(
    {
        "Flash_extent_density": (("y", "x"), temp_fed_lat_lon.Flash_extent_density_window.data)  # Original FED data
    },
    coords={
        "lat": (("y", "x"), temp_fed_lat_lon.lat.data),  # 2D latitude array
        "lon": (("y", "x"), temp_fed_lat_lon.lon.data)   # 2D longitude array
    }
)

# Now I'll interpolate it to the era5 grid. Note that the data itself has not changed yet, only the coordinates have. This transformation is accurate as it is the same
# as the NOAA provided coordinate transformation found at: https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php 

from scipy.interpolate import griddata

# Flatten the 2D lat, lon, and data without making a new copy of the arrays from the original dataset because scipy interpolate does not work unless the data is in 1D
lat_flat = fed_lat_lon.lat.values.ravel() 
lon_flat = fed_lat_lon.lon.values.ravel()
data_flat = fed_lat_lon.Flash_extent_density.values.ravel()

# Create a meshgrid of the target lat/lon on the ERA5 grid
target_lon, target_lat = np.meshgrid(era5_ds.longitude.values, era5_ds.latitude.values)

# Interpolate using  griddata
interpolated_data = griddata(
    points=(lon_flat, lat_flat),
    values=data_flat,
    xi=(target_lon, target_lat),
    method='linear'
)

# Wrap the result back into an xarray DataArray with ERA5 coordinates
interpolated_FED = xr.DataArray(
    interpolated_data,
    coords={
        "latitude": era5_ds.latitude,  # Match coordinate with era5 to ensure interpolation worked, if it didnt an error would occur here
        "longitude": era5_ds.longitude
    },
    dims=["latitude", "longitude"]
)





