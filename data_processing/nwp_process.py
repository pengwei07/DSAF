import pandas as pd
import numpy as np


ACCUMULATED_FEATURES = ['Clear-sky direct solar radiation at surface', 'Direct solar radiation',
                        'Downward UV radiation at the surface', 'Surface solar radiation downwards',
                        'Surface net solar radiation', 'Sunshine duration', 'Surface net solar radiation, clear sky',
                        'Surface thermal radiation downwards', 'Total sky direct solar radiation at surface',
                        'Top net thermal radiation', 'Top net solar radiation',
                        'Top net solar radiation, clear sky', 'Top net thermal radiation, clear sky',
                        'Total precipitation']

def process_accumulated_features(nwp_data, accumulated_features=ACCUMULATED_FEATURES):
    """
    Process accumulated nwp features.

    hour gap should be a constant array. Depends on ECMWF HRES data dissemination schedule:
    T+0 to T+90	        Hourly    	00 UTC, 06 UTC, 12 UTC and 18 UTC
    T+93 to T+144	    3-hourly	00 UTC and 12 UTC
    T+150h to T+240h	6-hourly	00 UTC and 12 UTC
    For more details, please refer https://www.ecmwf.int/en/forecasts/datasets/set-i#I-i-a_fc

    Args:
        nwp_data: DataFrame. Must be nwp data for a single site / coordinate from a dissemination run.
        accumulated_features: list
    """
    ## Convert accumulated features
    hour_gap = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

    processed_nwp_data = None
    if 'site' in nwp_data.columns:
        for site in nwp_data['site'].unique():
            cur_nwp_data = nwp_data[nwp_data['site'] == site].copy()
            cur_nwp_data.loc[:, 'gap'] = hour_gap[: cur_nwp_data.shape[0]]
            cur_nwp_data.loc[:, accumulated_features] = cur_nwp_data[accumulated_features].diff().div(cur_nwp_data['gap'].values * 3600, axis=0)
            cur_nwp_data.loc[cur_nwp_data.index[0], accumulated_features] = 0  # fill in np.nan with 0
            cur_nwp_data.drop(['gap'], axis=1, inplace=True)
            processed_nwp_data = pd.concat([processed_nwp_data, cur_nwp_data], axis=0)
    else:
        nwp_data['lat'] = nwp_data['lat'].round(1)
        nwp_data['lon'] = nwp_data['lon'].round(1)
        for lat in nwp_data['lat'].unique():
            for lon in nwp_data['lon'].unique():
                cur_nwp_data = nwp_data[(nwp_data['lat'] == lat) & (nwp_data['lon'] == lon)].copy()
                cur_nwp_data.loc[:, 'gap'] = hour_gap[: cur_nwp_data.shape[0]]
                cur_nwp_data.loc[:, accumulated_features] = cur_nwp_data[accumulated_features].diff().div(cur_nwp_data['gap'].values * 3600, axis=0)
                cur_nwp_data.loc[cur_nwp_data.index[0], accumulated_features] = 0  # fill in np.nan with 0
                cur_nwp_data.drop(['gap'], axis=1, inplace=True)
                processed_nwp_data = pd.concat([processed_nwp_data, cur_nwp_data], axis=0)

    return processed_nwp_data