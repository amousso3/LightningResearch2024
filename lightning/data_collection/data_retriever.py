import xarray as xr
import cdsapi
import netCDF4 as nc
from goes2go import GOES
import requests
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


# To retrieve the ERA5 data from the climate data store, use the cdsapi library.
# It requires you to make an account and to have an API key, which I have already done.

# There are two ERA5 datasets which we are interested in, reanalysis on single levels
# and reanalysis on pressure levels. The primary difference is that the data on pressure levels
# has an extra variable which refers to the pressure levels at which the data is obtained.

c = cdsapi.Client()
# To remove the need to memorize which variables are on single levels or not,
# this function will take in the variable nickname as input and verify the level type on its own.

single_lvl_variables = ['cape', 'tp']

# Cloud Cover, vertical velocity, temperature, cloud ice water content, speciic humiditiy
pressure_lvl_variables = ['cc', 'w', 't', 'ciwc', 'q'] 

time_list = [f"{str(hour).zfill(2)}:00" for hour in range(25)] # for if you want a day's worth of data

# Boundaries [N, W, S E]
def retrieve_era5(var: str, year: str, month: str,
                   day: str, times: str | list, boundaries: list[int], destination: str):
    if var in single_lvl_variables:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': ['reanalysis'],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'variable': [var],
                'year': [year],
                'month': [month],
                'day': [day],
                'time': [times],
                'area': boundaries,
            },
            destination)
    elif var in pressure_lvl_variables:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': ['reanalysis'],
                "data_format": "netcdf",
                "download_format": "unarchived",
                'variable': [var],
                'year': [year],
                'month': [month],
                'day': [day],
                'time': [times],
                'area': boundaries,
            },
            destination)
    else:
        print("The variable nickname provided is not documented.")
    return None


# Retrieving GOES GLM-LCFA or ABI-XXXX Data using goes2py
# There are two options, either download the data directly
# or produce a pandas dataframe of the data for a specific range of time
# Note that the data is recorded minutely.

variable_dictionary = {'LCFA': 'GLM-L2-LCFA', 'CTH': 'ABI-L2-ACHAF', 'CTT':'ABI-L2-ACHTF','Clear Sky Mask': 'ABI-L2-ACMF',
                        'Cloud Optical Depth': 'ABI-L2-CODF', 'Derived Stability Indicies': 'ABI-L2-DSIF',
                        'Land Surface Temp': 'ABI-L2-LSTF', 'Rainfall Rate Estimate': 'ABI-L2-RRQPEF'} # the keys will be the variable nicknames, and the values will be the product names

def retrieve_goes(var: str, start_time: str, end_time: str):
    G = GOES(satellite=16, product=variable_dictionary[var], domain='F')
    return xr.Dataset.from_dataframe(G.df(start= start_time, end= end_time))

#######################################################################################################################

# Retrieving the GOES-GLMF (Flash Extent Density variable)

# Function for downloading code using a HTTP GET request. If the file does not exist,
# it will return None, and skip that file.
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
    for minute in range(5, 60, 5):
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

    if hourly_sum is not None:
        hourly_sum = hourly_sum.expand_dims('time')
        hourly_sum['time'] = [datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour)]
    
    return hourly_sum, faulty_links

# Save the resulting hourly FED data to a NetCDF file
def save_fed_to_netcdf(fed_data, output_file):
    fed_data.to_netcdf(output_file)

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
        save_fed_to_netcdf(combined_fed, output_file)
        print(f"Saved 24-hour FED data to {output_file}")
    else:
        print("No valid data to save.")
    
    with open("faulty_links.txt", "w") as f:
        for link in all_faulty_links:
            f.write(f"{link}\n")

    print(f"Faulty links have been saved to faulty_links.txt")

