# Author: Maforikan Amoussou
# Date: 10/4/2024
# Description: Code for deriving important parameters such as:
# ICEFLUX, CAPE x Precip, Cloud Top Height, and Brightness Temperature

import xarray as xr
import numpy as np

def romps(cape, precip):
    return cape * precip

def ice_flux(q, w, t, s_h, c):
    """Returns the ICEFLUX parameterization of lightning from the 2014 Finney Study
        at 400 hPa.
    """
    p = 400*100 # Air Pressure
    R_v = 461.5 # Gas constant for water vapor
    R_d = 287.05 # Gas constant for Dry Air
    T_v = t * (1 + s_h * ((R_v/R_d) -1))
    rho = p / (R_d * T_v)
    F = w * rho # Updraught Mass Flux at 400 hPa
    phi = (q * F) /c #ICEFlUX Variable
    mask = c < 0.01
    phi = phi.where(~mask, 0)
    return phi