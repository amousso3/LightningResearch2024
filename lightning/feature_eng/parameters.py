# Author: Maforikan Amoussou
# Date: 10/4/2024
# Description: Code for deriving important parameters such as:
# ICEFLUX, CAPE x Precip, Cloud Top Height, and Brightness Temperature

import xarray as xr
import numpy as np
import cdsapi
import goes2go
import sys
import os
# Add the parent directory of 'lightning' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lightning.data_collection.data_retriever import retrieve_era5, retrieve_goes, single_lvl_variables, pressure_lvl_variables, time_list, retrieve_goes_glmf, variable_dictionary


# First we need to download the parameters from the climate data store and the goes2go package
retrieve_era5(single_lvl_variables, '2020', '04', '19', time_list, [45, -110, 25, -70], "/home/o/oneill/mamous3/lightning/datasets/cape_precip.nc")
retrieve_era5(pressure_lvl_variables, '2020', '04', '19', time_list, [45, -110, 25, -70], "/home/o/oneill/mamous3/lightning/datasets/ice_flux.nc")

