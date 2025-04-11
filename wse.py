
import netCDF4 as nc4;      import pandas as pd
import numpy as np;         import pathlib as pl
import requests;            import json

from sklearn.neighbors import BallTree


def noaa_data(begin,end,station,vdatum='NAVD',interval='6',
                       form='json',t_zone='GMT',unit='metric',product='water_level'):
    '''
    This function is used to get the data from NOAA API
    :param begin: begin date of the data
    :param end: end date of the data
    :param station: noaa station id
    :param vdatum: vertical datum such as NAVD
    :param interval: interval of the data (6 minutes)
    :param form: format of the data (json)
    :param t_zone: time zone of the data (GMT)
    :param unit: unit of the data (metric)
    :param product: type of the data (water_level)
    '''
    api = f'https://tidesandcurrents.noaa.gov/api/datagetter?begin_date={begin}&end_date={end}&station={station}'\
         f'&product={product}&application=NOS.COOPS.TAC.WL&datum={vdatum}&interval={interval}&time_zone={t_zone}&units={unit}&format={form}'
    data = requests.get(url=api).content.decode()
    return data


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def point_lookup(model_lat:np.array, model_lon:np.array, satellite_lat:np.array, satellite_lon:np.array):
    tree = BallTree(np.deg2rad(np.c_[model_lat,model_lon]), metric='haversine')
    distances, indices = tree.query(np.deg2rad(np.c_[satellite_lat, satellite_lon]), k = 1)
    return distances*6371,indices

def extract_model_wse_for_station(name, stations, years, root, resample_time='D'):
    """
    Extract model water surface elevation (WSE) for a given station name across multiple years.
    
    Parameters:
        name (str): Station name key from the stations dictionary
        stations (dict): Dictionary with station lat/lon info
        years (list): List of year strings (e.g., ['2019', '2020', '2021'])
        root (Path): Root path to model output directory
        resample_time (str): Resample frequency (e.g., 'D' for daily)
        
    Returns:
        df_daily_wse (DataFrame): Daily max WSE with Year and Month columns
        df_hourly_wse (DataFrame): Hourly WSE
        df_annual_max (DataFrame): Annual max WSE and trend
    """
    obs_lat = stations[name]['lat']
    obs_lon = stations[name]['lon']
    df_list, df_hourly_list = [], []

    for year in years:
        ncfile = nc4.Dataset(root / year / 'outputs' / 'fort.63.nc')
        start = pd.to_datetime(ncfile.variables['time'].base_date)
        dt = pd.date_range(start=start, freq='1h', periods=ncfile.dimensions['time'].size)
        
        x, y = ncfile.variables['x'][:], ncfile.variables['y'][:]
        distances, node_ids = point_lookup(y, x, obs_lat, obs_lon)
        node_id = node_ids[0][0]

        model = ncfile.variables['zeta'][:, node_id]
        df = pd.DataFrame({'dt': dt, 'data': model})

        df_daily_max = df.resample(resample_time, on='dt').max().reset_index()
        df_daily_max["Year"] = df_daily_max["dt"].dt.year
        df["Year"] = df["dt"].dt.year  # Needed for filtering later

        df_list.append(df_daily_max)
        df_hourly_list.append(df)

    # Combine daily and hourly data
    df_daily_wse = pd.concat(df_list, ignore_index=True).iloc[1:].reset_index(drop=True)
    df_hourly_wse = pd.concat(df_hourly_list, ignore_index=True)

    # Add Year/Month to daily df
    df_daily_wse["Year"] = df_daily_wse["dt"].dt.year
    df_daily_wse["Month"] = df_daily_wse["dt"].dt.month

    # Compute annual max and linear trend
    df_annual_max = df_daily_wse.groupby("Year")["data"].max().reset_index()
    slope_annual, intercept_annual, r_value, p_value, std_err = linregress(df_annual_max["Year"], df_annual_max["data"])
    df_annual_max["Expected_WSE"] = df_annual_max["Year"] * slope_annual + intercept_annual

    return df_daily_wse, df_hourly_wse, df_annual_max

def extract_model_wse_and_ice(name, stations, years, root):
    """
    Extract hourly and daily model WSE and sea ice concentration for a station over multiple years.

    Parameters:
        name (str): Station key from `stations` dictionary
        stations (dict): Dictionary with station lat/lon info
        years (list of str): List of year strings (e.g., ['2020', '2021'])
        root (Path): Path object pointing to the root directory with yearly output folders

    Returns:
        df_all (DataFrame): Hourly data for all years (datetime, WSE, ice concentration)
        df_daily (DataFrame): Daily resampled WSE max and ice concentration mean
    """
    obs_lat = stations[name]['lat']
    obs_lon = stations[name]['lon']
    df_list = []

    for year in years:
        try:
            # --- Water level file (fort.63.nc) ---
            ncfile_wse = nc4.Dataset(root / year / 'outputs' / 'fort.63.nc')
            start_time = pd.to_datetime(ncfile_wse.variables['time'].base_date)
            dt = pd.date_range(start=start_time, freq='1h', periods=ncfile_wse.dimensions['time'].size)

            x = ncfile_wse.variables['x'][:]
            y = ncfile_wse.variables['y'][:]

            # Find nearest node
            distances, node_ids = point_lookup(y, x, obs_lat, obs_lon)
            node_id = node_ids[0][0]

            model_wse = ncfile_wse.variables['zeta'][:, node_id]

            # --- Ice file (fort.93.nc) ---
            ncfile_ice = nc4.Dataset(root / year / 'outputs' / 'fort.93.nc')
            ice_conc = ncfile_ice.variables['iceaf'][:, node_id]

            # --- Combine into DataFrame for this year ---
            df_year = pd.DataFrame({
                'dt': dt,
                'model_wse': model_wse,
                'ice_conc': ice_conc
            })
            df_year["year"] = int(year)
            df_list.append(df_year)

        except Exception as e:
            print(f"Error processing year {year}: {e}")

    # --- Combine all years ---
    df_all = pd.concat(df_list, ignore_index=True)

    # --- Resample to daily: max WSE, mean ice concentration ---
    df_daily = df_all.resample("D", on="dt").agg({
        "model_wse": "max",
        "ice_conc": "mean"
    }).reset_index()

    return df_all, df_daily