#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Encoding support for wildlife computers netCDF."""
#####
##To run this code is necesary to install eccodes
## sudo apt-get install libeccodes-dev
# sudo apt-get install libeccodes-tools
# sudo apt-get install python-pip
# pip install eccodes-python
# python -m eccodes selfcheck

import numpy as np
import xarray as xr
import sys
import traceback
from eccodes import *

def pres(depth, lat):
    """
    Calculates pressure in dbars from depth in meters.
    Parameters
    ----------
    depth : array_like
            depth [meters]
    lat : array_like
          latitude in decimal degrees north [-90..+90]
    Returns
    -------
    p : array_like
           pressure [db]
    Examples
    --------
    >>> import seawater as sw
    >>> depth, lat = 7321.45, 30
    >>> sw.pres(depth,lat)
    7500.0065130118019
    References
    ----------
    .. [1] Saunders, Peter M., 1981: Practical Conversion of Pressure to Depth.
       J. Phys. Oceanogr., 11, 573-574.
       doi: 10.1175/1520-0485(1981)011<0573:PCOPTD>2.0.CO;2
    """
    depth, lat = list(map(np.asanyarray, (depth, lat)))
    deg2rad = np.pi / 180.0
    X = np.sin(np.abs(lat * deg2rad))
    C1 = 5.92e-3 + X ** 2 * 5.25e-3
    return ((1 - C1) - (((1 - C1) ** 2) - (8.84e-6 * depth)) ** 0.5) / 4.42e-6

profile_dataset='/data/obs/mangopare/MOANA_0230_342_230306230658_qc.nc'
class BUFR():
    def __init__(
            self,
            profile_filename,
            output_filename,
            directory
    ):
        self.profile_filename=profile_filename
        self.output_filename=output_filename
        self.directory=directory
        self.path= 
    
    def create_variables_from_netcdf(self, profile_filename):
        self.ds=xr.open_dataset(profile_filename)
        self.df = self.ds.to_dataframe()
        QC=np.where(self.df['QC_FLAG']==1)[0]
        self.df=self.df.iloc[QC]
        self.df['PRESSURE']=pres(self.df['DEPTH'],self.df['LATITUDE'])*10000
        self.years  = self.df.index.year.values
        self.months = self.df.index.month.values
        self.days   = self.df.index.day.values
        self.hours  = self.df.index.hour.values 
        self.minutes= self.df.index.minute.values
        self.seconds= self.df.index.second.values
        self.latitudes    = self.df["LATITUDE"].values
        self.longitudes   = self.df["LONGITUDE"]
        self.pressures    = np.round(self.df["PRESSURE"],2)
        self.temperatures = self.df["TEMPERATURE"]+273.15
        
    def create_bufr_file(self, output_filename):
        VERBOSE = 1  # verbose error reporting
        ibufr = codes_bufr_new_from_samples("BUFR3_local")
        #######################################
        #########Section 1, Header ############
        #######################################
        codes_set(ibufr, "edition", 3)
        codes_set(ibufr, "masterTableNumber", 0)
        codes_set(ibufr, "bufrHeaderSubCentre", 0)
        codes_set(ibufr, "bufrHeaderCentre", 69) #MetService Centre code from table Code Table C-11 69 -> Wellington (RSMC)
        codes_set(ibufr, "updateSequenceNumber", 0)
        codes_set(ibufr, "dataCategory", 31) # CREX Table A 31 -> Oceanographic Data 
        #codes_set(ibufr, "dataSubCategory", 182) #International data-subcategory 
        codes_set(ibufr, "masterTablesVersionNumber", 28) #Latest version 28 -> 15 November 2021
        codes_set(ibufr, "localTablesVersionNumber", 0)
        codes_set(ibufr, "typicalYearOfCentury", 23)
        codes_set(ibufr, "typicalMonth", int(self.months[0]))
        codes_set(ibufr, "typicalDay", int(self.days[0]))
        codes_set(ibufr, "typicalHour", int(self.hours[0]))
        codes_set(ibufr, "typicalMinute", int(self.minutes[0]))
        codes_set(ibufr, "ident", self.ds.moana_serial_number) # Moana Serial Number
        codes_set(ibufr, "numberOfSubsets", len(self.df))
        codes_set(ibufr, "observedData", 1)
        codes_set(ibufr, "compressedData", 1)

        codes_set(ibufr, "inputDelayedDescriptorReplicationFactor",len(self.df)) ##Each observation in each file is a subset 
        #The codes that are replicated are the following 
        #315003 Data itself
        #301011 Time 
        #005001 Latitude
        #006001 Longitude
        #008080 Qualifier for GTSPP Quality Flag
        #033050 Global GTSPP Quality Flag
        #007065 Water Pressure (Pa)
        #008080 Qualifier for GTSPP Quality Flag
        #033050 Global GTSPP Quality Flag
        #022045 Oceanographic Water Temperature (K)
        codes_set_array(
            ibufr,
            "unexpandedDescriptors",
            (315003, 301011, 5001, 6001, 8080, 33050, 7065, 8080, 33050, 22045),
        )
        # Create the structure of the data section
        codes_set(ibufr, "marineObservingPlatformIdentifier", int(self.ds.moana_serial_number))
        codes_set(ibufr, "observingPlatformManufacturerModel", 'Mangopare')
        codes_set(ibufr, "observingPlatformManufacturerSerialNumber", self.ds.deck_unit_serial_number)
        codes_set(ibufr, "buoyType", 2)#CODE-Table 2 -> subsurface float, moving
        codes_set(ibufr, "dataCollectionLocationSystem", 2)#CODE_Table 2 -> GPS
        codes_set(ibufr, "dataBuoyType",8)#Code-Table  8 -> Unspecified subsurface float
        codes_set(ibufr, "directionOfProfile", 3) #Code-Table 3-> Missing value
        codes_set(ibufr, "instrumentTypeForWaterTemperatureOrSalinityProfileMeasurement", 995)
        #codes_set(ibufr, "Year", years)
        codes_set_array(ibufr, "#1#waterPressure", self.pressures)
        codes_set(ibufr, "#1#QualifierForGTSPPQualityFlag", 10)
        codes_set(ibufr, "#1#GlobalGTSPPQualityFlag", 9)
        codes_set_array(ibufr, "#1#year", self.years)
        codes_set_array(ibufr, "#1#month", self.months)
        codes_set_array(ibufr, "#1#day", self.days)
        codes_set_array(ibufr, "#1#hour", self.hours)
        codes_set_array(ibufr, "#1#minute", self.minutes)
        codes_set_array(ibufr, "#1#latitude", self.latitudes)
        codes_set_array(ibufr, "#1#longitude",self.longitudes)
        codes_set_array(ibufr, "#1#oceanographicWaterTemperature", self.temperatures)
        codes_set(ibufr, "#2#QualifierForGTSPPQualityFlag", 11)
        codes_set(ibufr, "#2#GlobalGTSPPQualityFlag", 9)
        codes_set(ibufr, "#1#salinity", CODES_MISSING_DOUBLE)
        # Encode the keys back in the data section
        codes_set(ibufr, "pack", 1)
        # Create output file 
        outfile = open(output_filename, "wb")
        codes_write(ibufr, outfile)
        print("Created output BUFR file ",output_filename)
        codes_release(ibufr)

    def run(self):
        self.create_bufr_file(self.profile_filename)
        self.create_variables_from_netcdf(self.output_filename)