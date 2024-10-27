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
# Important Note: Lat/Lon data is usable unless you take a subset in an area further away from the boundaries due to NaNs

