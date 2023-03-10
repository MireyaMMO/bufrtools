#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Encoding support for wildlife computers netCDF."""


import io
import sys
from typing import List
from pathlib import Path
from argparse import Namespace, ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd

from bufrtools.tables import get_sequence_description
from bufrtools.encoding import bufr as encoder
from bufrtools.util.gis import azimuth, haversine_distance
from bufrtools.util.parse import parse_input_to_dataframe

import xarray as xr 
import importlib as il

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

def get_section1() -> dict:
    """Returns the section1 part of the message to be encoded."""
    now = datetime.utcnow()
    section1 = {
        'originating_centre': 0, #MetOcean doesn't have originatin centre 
        'sub_centre': 0,
        'data_category': 31,         # oceanographic data
        'sub_category': 4,           # subsurface float (profile)
        'local_category': 0,         # Ideally something specifies this as a marine mammal
                                     # animal tag
        'master_table_version': 39,
        'local_table_version': 255,  # Unknown
        'year': now.year,
        'month': now.month,
        'day': now.day,
        'hour': now.hour,
        'minute': now.minute,
        'second': now.second,
        'seq_no': 0,                 # Original message
    }
    return section1


def get_section3() -> dict:
    """Returns the section3 part of the message to be encoded."""
    section3 = {
        'number_of_subsets': 1,
        'observed_flag': True,
        'compressed_flag': False,
        'descriptors': ['315023'],
    }
    return section3

def get_mangopare_data(df: pd.DataFrame) -> List[dict]:
    """Returns a sequence of records for the trajectory part of the BUFR message."""
    # Pull profile locations out as the first point in each profile
    QC=np.where(df['QC_FLAG']==1)[0]
    dataset = pd.DataFrame({
        'time': df.index[QC],
        'temperature': df['TEMPERATURE'][QC],
        'lat': df['LATITUDE'][QC],
        'lon': df['LONGITUDE'][QC],
  #      'depth':df['DEPTH'][QC]
    })
    dataset['pressure']=pres(df['DEPTH'][QC],df['LATITUDE'][QC])*1000
    # Should we ever have a negative depth?
    #trajectory['z'] = trajectory.z.apply(lambda x: max(0, x))
    # Drop the profile index before merge
    dataset = dataset.reset_index(drop=True)
    # Combine back with the full dataset after calculating
    # speed and direction
    sequence = []
    sequence.append({
        'fxy': '031001',
        'text': 'Delayed descriptor replication factor (Numeric)',
        'type': 'numeric',
        'scale': 0,
        'offset': 0,
        'bit_len': 8,
        'value': len(dataset)
    })
    data_seq = get_sequence_description('306048').iloc[21::]
    for _, row in dataset.iterrows():
        for seq in process_mangopare(data_seq.copy(), row):
            sequence.append(seq)
    return sequence


def process_mangopare(data_seq: pd.DataFrame, row) -> List[dict]:
    """Returns the sequence for the given row of the trajectory data frame."""
    
    # Get temperature
    temperature = getattr(row, 'temperature', np.nan)
    temperature += 273.15  # Convert from deg_C to Kelvin
    
    data_seq['value'] = [
        1,                           # Time significance aka Time series 
        np.nan,                      # Sequence
        row.time.year,
        row.time.month,
        row.time.day,
        np.nan,                      # Sequence
        row.time.hour,
        row.time.minute,
        1,                           # Coordinates significance; 1=Observation coordinates 
        np.nan,                      # Lat/Lon Sequence,
        row.lat,
        row.lon,
        1,                           # Global quality flag 1=Correct value
        row.pressure,                # Depth (pressure)
        10,                          # Quality flag, 10=Water pressure at level
        9,                           # Global quality flag 9=Good for operational use
        temperature,                 # Sea / Water Temperature (K)
        11,                          # Quality flag, 10=Water temperature at level
        9,                           # Global quality flag 9=Good for operational use
        ]
    return data_seq.to_dict(orient='records')


def get_section4(df: pd.DataFrame, **kwargs) -> List[dict]:
    """Returns the section4 data."""
    records = []

    wigos_issuer = int(kwargs.pop('wigos_issuer', 22000))
    wigos_local_identifier = str(kwargs.pop('wigos_platform_code', ''))
    wigos_identifier_series = 0  # Placeholder
    wigos_issue_number = 0       # Placeholder

    wigos_sequence = get_sequence_description('301150')
    wigos_sequence['value'] = [
        np.nan,                   # Sequence
        wigos_identifier_series,  # 001125,WIGOS identifier series,,,Operational
        wigos_issuer,             # 001126,WIGOS issuer of identifier,,,Operational
        wigos_issue_number,       # 001127,WIGOS issue number,,,Operational
        wigos_local_identifier,   # 001128,WIGOS local identifier (character),,,Operational
    ]
    records.extend(wigos_sequence.to_dict(orient='records'))
    wmo=0
 #   uuid = kwargs.pop('uuid')
 #   ptt = kwargs.pop('ptt')
 #   wmo = kwargs.pop('wmo_platform_code', None)
    # If WMO ID is passed in as None, fill it with zero
    if wmo is None:
        wmo = 0

    platform_id_sequence = get_sequence_description('306048')[6:19]
    platform_id_sequence['value'] = [
        np.nan,         # 201129,Change data width,,,Operational  # noqa
        0,              # 001087,WMO marine observing platform extended identifier ,WMO number where assigned,,Operational # noqa
        np.nan,         # 201000,Change data width,Cancel,,Operational
        np.nan,         # 208032,Change width of CCITT IA5 ,change width to 32 characters,,Operational # noqa
        '63755LL',      # 001019,Ship or mobile land station identifier ,"vessel_id (max 32 characters)",,Operational # noqa
        np.nan,         # 208000,Change width of CCITT IA5 ,Cancel change width,,Operational # noqa
        11,             # 003001,Surface station type , 11 vessel moving sensor, Operational # noqa
        997,            # 022067,Instrument type for water temperature and/or salinity measurement,set to 997 Ocean moving vessel profiler),,Operational # noqa
        '5291',         # 001051,Platform transmitter ID number,deck_unit_serial_number,,Operational # noqa
        11,             # 002148,Data collection and/or location system,,,Operational # noqa New MANGOPARE SENSOR or 2 for GPS
        230,            # 001154,Sensor ID, Moana Serial Number
        0,              # 008015,Significant qualifier for sensor,,, Set to 0 for single senor
        4.17            # 025026,Battery Voltage,,deck_unit_battery_voltage
    ]
    records.extend(platform_id_sequence.to_dict(orient='records'))
    # WC profiles don't have enough data to fill in the trajectory portion of the BUFR, so we'll
    records.extend(get_mangopare_data(df))
    return records


def encode(profile_dataset: Path, output: Path, **kwargs):
    """Encodes the input `profile_dataset` as BUFR and writes it to `output`."""
 #   df, meta = parse_input_to_dataframe(profile_dataset)
    ds=xr.open_dataset(profile_dataset)
    dfn = ds.to_dataframe()
    # If we were able to extract metadata attributes from the
    # source dataset, use those instead of the passed in values
    if meta:
        kwargs = {**kwargs, **meta}

    context = {}
    context['buf'] = buf = io.BytesIO()
    encoder.encode_section0({}, context)
    section1 = get_section1()
    encoder.encode_section1({'section1': section1}, context)
    section3 = get_section3()
    encoder.encode_section3({'section3': section3}, context)
    section4 = get_section4(df, **kwargs)
    encoder.encode_section4({'section4': section4}, context)
    encoder.encode_section5(context)
    encoder.finalize_bufr(context)

    buf.seek(0)
    output.write_bytes(buf.read())


def parse_args(argv) -> Namespace:
    """Returns the namespace parsed from the command line arguments."""
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument('-o',
                        '--output',
                        type=Path,
                        default=Path('output.bufr'), help='Output file')
    parser.add_argument('profile_dataset', type=Path, help='ATN Wildlife Computers profile netCDF')
    parser.add_argument('-u',
                        '--uuid',
                        type=str,
                        default=None)
    parser.add_argument('-p',
                        '--ptt',
                        type=str,
                        default=None)

    args = parser.parse_args(argv)
    return args


def main():
    """Encode a wildlife computers profile."""
    args = parse_args(sys.argv[1:])

    assert args.profile_dataset.exists()
    encode(args.profile_dataset, args.output, uuid=args.uuid, ptt=args.ptt)
    return 0


if __name__ == '__main__':
    sys.exit(main())


profile_dataset='/data/obs/mangopare/MOANA_0230_342_230306230658_qc.nc'