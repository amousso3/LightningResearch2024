# Author: Maforikan Amoussou
# Date: 10/15/2024
# Description: In this file I will validate data from the data sources
# by ensuring that (a) FED Window = FED Summed Over 5 Minutes 
# (b) coordinate transformation is accurate and (c) comparing ERA5 Products
# to their ABI versions

# FED Window vs FED summed over 5 minutes
# So I need 5 files for any given 5 minute interval
# I need to verify that summing the FED value for all
# 5 minutes of the interval is equal to the last minute's FED window

import xarray as xr
import netCDF4 as nc
import pyproj
import numpy as np
import requests
import os

# URL for the file
urls = ["https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/2023/2023001/OR_GLM-L2-GLMF-M3_G16_e20230101000000.nc",
        "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/2023/2023001/OR_GLM-L2-GLMF-M3_G16_e20230101000100.nc",
        "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/2023/2023001/OR_GLM-L2-GLMF-M3_G16_e20230101000200.nc",
        "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/2023/2023001/OR_GLM-L2-GLMF-M3_G16_e20230101000300.nc",
        "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/2023/2023001/OR_GLM-L2-GLMF-M3_G16_e20230101000400.nc",
        "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/2023/2023001/OR_GLM-L2-GLMF-M3_G16_e20230101000500.nc"]

output_path = "/home/o/oneill/mamous3/lightning/datasets"  # Replace with your desired save path
# Download the file
'''for url in urls:
    download_path = output_path + url[-41:]
    if (os.path.isdir(download_path)):
        response = requests.get(url)
        with open(download_path, "wb") as file:
            file.write(response.content)

        print("Download complete.")'''


FED_minute_1 = xr.open_dataset(output_path + urls[0][-41:])
FED_minute_2 = xr.open_dataset(output_path + urls[1][-41:])
FED_minute_3 = xr.open_dataset(output_path + urls[2][-41:])
FED_minute_4 = xr.open_dataset(output_path + urls[3][-41:])
FED_minute_5 = xr.open_dataset(output_path + urls[4][-41:])
FED_minute_6 = xr.open_dataset(output_path + urls[5][-41:])

datasets = [FED_minute_1, FED_minute_2, FED_minute_3, FED_minute_4, FED_minute_5]
FED_sum = sum(ds['Flash_extent_density'] for ds in datasets)

print(FED_sum == FED_minute_5['Flash_extent_density'])
# Conclusion: Inconclusive solely via verifying equality due to nan's and due to potential smoothing effects in the FED window not present in the minutely view
# However, it is logical to assume, based on the definition of the FED window parameter, that integrated and aggregated can be interchanged in this context, so
# summing the FED window over the hour should be a valid method of temporally coarsening the dataset.

# (b) Verifying the coordinate transformation from x and y to latitude and longitude.
# Approach 1: Mapping the data visually (Already did this for April 20th)
# Comparing the location of data with spots you know should have data will clue us into
# the accuracy of the transformation, however, since this data is huge, we won't know how accurate it actually is

# Approach 2: Converion and Undoing Conversion
# The undone conversion should be close if not exactly the same as the original dataset,
# This may also help us quantify the error propagation that comes as a result of the coordinate transformation

