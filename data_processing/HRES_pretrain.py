# Example, showing how to process the HRES dataset

import datetime
import pandas as pd
import pygrib
import time
from pathlib import Path
import json
from utils.nwp_process import process_accumulated_features
from utils.tools import get_full_grids

SURFACE_FEATURES = [
                    'Surface pressure',
                    '10 metre U wind component',
                    '10 metre V wind component',
                    '2 metre temperature',
                    '2 metre dewpoint temperature',
                    'Skin temperature',
                    '100 metre U wind component',
                    '100 metre V wind component',
                    ]


# @check_input_dict
def parse_grb_file(file_path: str, area_idx, features_to_use, site_list, default_valid_date=None, ):
    """
    Parse single grib file from *Single Level* using dict.
    For each stepRange, the data will be flattened into lists, one for each feature,
    and merge with the previous dict.

    Args:
        file_path: path to grib file.
        area_idx: list of tuples (lat_max, lat_min, lon_max, lon_min).
        features_to_use: None or list. Default is None.
        default_valid_date: None or given.
    Returns: dict of the form {feature_name: [data]}.
    """

    grb = pygrib.open(file_path)
    data = grb.read()

    if not data:
        print(f'data file {file_path} is empty, skipping the file ...')
        return {}

    lat, lon = grb.message(1).latlons()
    lat = lat[:, 0]
    lon = lon[0, :]

    lat = lat[area_idx[0]].flatten().tolist()
    lon = lon[area_idx[1]].flatten().tolist()

    num_areas = len(lat)
    curr_fcst_time = None

    if site_list is not None:
        cleaned_data = {'fcst_date': [], 'lat': [], 'lon': [], 'valid_date': [], 'site': [], 'site_id': []}
    else:
        cleaned_data = {'fcst_date': [], 'lat': [], 'lon': [], 'valid_date': []}

    valid_date_mismatch = 0
    for attr in data:

        if features_to_use is not None and attr.name not in features_to_use:
            continue

        try:
            fcst_time = int(attr['stepRange'])
        except ValueError:
            fcst_time = int(attr['stepRange'].split('-')[-1])
            logger.warning('step range: {} is a range, using {}'.format(attr['stepRange'], fcst_time))

        attr_value = attr.values[area_idx[0], area_idx[1]].flatten().tolist()

        if attr.name not in cleaned_data.keys():
            cleaned_data[attr.name] = []

        if curr_fcst_time != fcst_time:
            curr_dt = [default_valid_date + datetime.timedelta(hours=fcst_time)] * num_areas
            valid_dt = [default_valid_date] * num_areas
            cleaned_data['fcst_date'].extend(curr_dt)
            cleaned_data['valid_date'].extend(valid_dt)
            cleaned_data['lat'].extend(lat)
            cleaned_data['lon'].extend(lon)
            if site_list is not None:
                cleaned_data['site'].extend(site_list[0])
                cleaned_data['site_id'].extend(site_list[1])

            curr_fcst_time = fcst_time

        cleaned_data[attr.name].extend(attr_value)

    return cleaned_data


def parse_grb_file_pressure_level(file_path: str,
                                  area_idx,
                                  altitude_features_to_use,
                                  pressure_levels,
                                  site_list,
                                  default_valid_date=None, ):
    """
    Parse single grib file from *Pressure Level* using dict.
    For each stepRange, the data will be flattened into lists, one for each feature,
    and merge with the previous dict.

    Args:
        file_path: path to grib file.
        area_idx: list of tuples (lat_max, lat_min, lon_max, lon_min).
        features_to_use: None or list. Default is None.
        default_valid_date: None or given.
    Returns: dict of the form {feature_name: [data]}.
    """

    grb = pygrib.open(file_path)
    data = grb.read()

    if not data:
        print(f'data file {file_path} is empty, skipping the file ...')
        return {}

    lat, lon = grb.message(1).latlons()
    lat = lat[:, 0]
    lon = lon[0, :]

    lat = lat[area_idx[0]].flatten().tolist()
    lon = lon[area_idx[1]].flatten().tolist()

    num_areas = len(lat)
    curr_fcst_time = None

    if site_list is not None:
        cleaned_data = {'fcst_date': [], 'lat': [], 'lon': [], 'valid_date': [], 'site': [], 'site_id': []}
    else:
        cleaned_data = {'fcst_date': [], 'lat': [], 'lon': [], 'valid_date': []}

    valid_date_mismatch = 0

    for attr in data:

        if attr.name not in altitude_features_to_use or attr.level not in pressure_levels:
            continue
        feature_name = (f'{attr.name} {attr.level}').replace(' ', '_')
        feature_name += 'hPa'

        try:
            fcst_time = int(attr['stepRange'])
        except ValueError:
            fcst_time = int(attr['stepRange'].split('-')[-1])
            logger.warning('step range: {} is a range, using {}'.format(attr['stepRange'], fcst_time))

        attr_value = attr.values[area_idx[0], area_idx[1]].flatten().tolist()
        if feature_name not in cleaned_data.keys():
            cleaned_data[feature_name] = []
        if curr_fcst_time != fcst_time:
            curr_dt = [default_valid_date + datetime.timedelta(hours=fcst_time)] * num_areas
            valid_dt = [default_valid_date] * num_areas
            cleaned_data['fcst_date'].extend(curr_dt)
            cleaned_data['valid_date'].extend(valid_dt)
            cleaned_data['lat'].extend(lat)
            cleaned_data['lon'].extend(lon)
            if site_list is not None:
                cleaned_data['site'].extend(site_list[0])
                cleaned_data['site_id'].extend(site_list[1])
            curr_fcst_time = fcst_time

        cleaned_data[feature_name].extend(attr_value)

    return cleaned_data


def get_raw_ec_df(file_path,
                  nwp_path,
                  site_info,
                  surface_features,
                  pressure_level_features,
                  pressure_levels,
                  preprocess=False,
                  ):
    nwp_path = Path(nwp_path) if isinstance(nwp_path, str) else nwp_path
    nwp_path.mkdir(exist_ok=True, parents=True)

    selected_info = [
        [k,
         site_info[k].get('site_id', k),
         eval(site_info[k]['area_idx'])[0],
         eval(site_info[k]['area_idx'])[2],
         ] \
        for k in site_info.keys()]

    site_info = pd.DataFrame(selected_info, columns=['site', 'site_id', 'lat_idx', 'lon_idx'])

    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    nwp_path = Path(nwp_path) if isinstance(nwp_path, str) else nwp_path

    for valid_date_path in sorted(file_path.glob('20*')):
        default_valid_date = pd.to_datetime(valid_date_path.name, format='%Y%m%d%H').strftime('%Y%m%d_%H%S')
        if len(list(Path(valid_date_path / 'surface').glob('*grib'))) == 0:
            continue

        final_data_surface, final_data_pressure_level = None, None

        for f in sorted(Path(valid_date_path / 'surface').glob('*.grib')):
            cur_data_surface = parse_grb_file(str(f),
                                              [site_info['lat_idx'].values, site_info['lon_idx'].values],
                                              surface_features,
                                              (site_info['site'].values, site_info['site_id'].values),
                                              pd.to_datetime(default_valid_date, format='%Y%m%d_%H%M'))
            cur_data_surface = pd.DataFrame(cur_data_surface)
            final_data_surface = pd.concat([final_data_surface, cur_data_surface], axis=0)

        for f in sorted(Path(valid_date_path / 'pressure_level').glob('*.grib')):
            cur_data_pressure = parse_grb_file_pressure_level(str(valid_date_path / 'pressure_level' / f.name),
                                                              [site_info['lat_idx'].values,
                                                               site_info['lon_idx'].values],
                                                              pressure_level_features,
                                                              pressure_levels,
                                                              (site_info['site'].values, site_info['site_id'].values),
                                                              pd.to_datetime(default_valid_date, format='%Y%m%d_%H%M'))
            cur_data_pressure = pd.DataFrame(cur_data_pressure)
            final_data_pressure_level = pd.concat([final_data_pressure_level, cur_data_pressure], axis=0)

        final_data = pd.merge(final_data_surface,
                              final_data_pressure_level.drop(['lat', 'lon'], axis=1),
                              on=['fcst_date', 'valid_date', 'site', 'site_id'])
        final_data = final_data.sort_values(by='fcst_date')

        if preprocess is True:
            final_data['fcst_date'] = pd.to_datetime(final_data['fcst_date'])
            final_data['valid_date'] = pd.to_datetime(final_data['valid_date'])
            final_data['fcst_date'] = final_data['fcst_date'] + pd.Timedelta(8, unit='hour')
            final_data = final_data.drop_duplicates(subset=['fcst_date', 'lon', 'lat', 'site'], keep='first')
            final_data = process_accumulated_features(final_data)

            numerical_cols = set(final_data.columns.values)
            numerical_cols = numerical_cols.difference(set(['valid_date']))

            nwp_all = None
            for site in set(final_data['site'].values):
                cur_nwp = final_data[final_data['site'] == site]

                cur_nwp = cur_nwp[numerical_cols].set_index('fcst_date').resample(rule='15T', closed='right',
                                                                                  label='right').interpolate(
                    method='linear')
                cur_nwp = cur_nwp.ffill().bfill()
                # cur_nwp['site'] = site
                nwp_all = pd.concat([nwp_all, cur_nwp])
            final_data = nwp_all

        final_data.reset_index().to_csv(nwp_path / f'{default_valid_date}.csv.gz',
                                        index=False,
                                        compression='gzip',
                                        encoding='gbk')


def get_raw_ec_df_by_area(file_path, nwp_path, coord_info, features_to_use, ):
    nwp_path = Path(nwp_path) if isinstance(nwp_path, str) else nwp_path
    nwp_path.mkdir(exist_ok=True, parents=True)
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    nwp_path = Path(nwp_path) if isinstance(nwp_path, str) else nwp_path

    for valid_date_path in sorted(file_path.glob('20*')):

        default_valid_date = pd.to_datetime(valid_date_path.name[:13], format='%Y%m%d_%H%M').strftime('%Y%m%d_%H%M')

        # if len(list(Path(valid_date_path / 'surface').glob('*grib'))) == 0:
        #     continue

        final_data = None
        for f in sorted((valid_date_path / 'surface').glob('*.grib')):
            cur_data = parse_grb_file(str(f),
                                      coord_info,
                                      features_to_use,
                                      None,
                                      pd.to_datetime(default_valid_date, format='%Y%m%d_%H%M'))

            cur_data = pd.DataFrame(cur_data)
            final_data = pd.concat([final_data, cur_data], axis=0)

        final_data = final_data.sort_values(by='fcst_date')
        # final_data.to_csv(nwp_path / f'{default_valid_date}.csv.gz', index=False, compression='gzip', encoding='gbk')
        final_data.to_csv(nwp_path / f'{default_valid_date}.csv', index=False, encoding='gbk')


if __name__ == '__main__':
    file_path = 'EC_data_processing/huadong/'
    nwp_path = 'EC_data_processing/huadong_history/'

    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    nwp_path = Path(nwp_path) if isinstance(nwp_path, str) else nwp_path

    coords = get_full_grids((0, 161, 0, 111))

    files = sorted(file_path.glob('*.grib'))
    for file in files:
        valid_date = file.name[15:28]
        file = file_path / file
        data = parse_grb_file(str(file), coords, SURFACE_FEATURES, None,
                              pd.to_datetime(valid_date, format='%Y%m%d_%H%M'))
        data = pd.DataFrame(data)
        data = data.sort_values(by='fcst_date')
        data.to_csv(nwp_path / f'{valid_date}.csv', index=False, encoding='gbk')
