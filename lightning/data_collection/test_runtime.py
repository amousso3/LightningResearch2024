from data_retreiver_2 import retrieve_goes_glmf
import xarray as xr
import cdsapi
import netCDF4 as nc
from goes2go import GOES
import requests
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

target = xr.open_dataset('49532bf0b60c575c14ab777b16ea9b54.nc')

date = datetime(2023, 7, 1)
boundaries = [(25, 45), (-110, -70)]
retrieve_goes_glmf(date, 'glm_time_test.nc', boundaries, target)

import time
# Define tests
tests = {
    "30 days": 30,
}

# Initialize results
results = {}

# Run tests
for test_name, days in tests.items():
    start_date = datetime(2023, 7, 1)
    total_runtime = 0

    for i in range(days):
        date = start_date + timedelta(days=i)
        start_time = time.time()
        
        # Call the function for each day
        retrieve_goes_glmf(date, f"glm_time_{date.strftime('%Y%m%d')}.nc", boundaries, target)
        
        end_time = time.time()
        total_runtime += (end_time - start_time)

    # Convert total runtime to minutes
    total_runtime_minutes = total_runtime / 60
    results[test_name] = total_runtime_minutes
    print(f"{test_name}: {total_runtime_minutes:.2f} minutes")
    

# Save results to a file
with open("function_runtimes.txt", "w") as f:
    for test, runtime in results.items():
        f.write(f"{test}: {runtime:.2f} minutes\n")
