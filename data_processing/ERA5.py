# Example, showing how to process the ERA5 dataset
import pandas as pd 
import numpy as np 
import datetime 
import os
import pygrib

def get_full_grids(area_idx):
    # N,S,W,E = eval(area idx[0]), eval(area idx[1]), eval(areaidx[2]),eval(area idx[3])
    N,S,W,E = area_idx
    lat_coord = np.arange(N,(S + 1))
    lon_coord = np.arange(W,(E + 1))
    all_coords = [[i, j] for i in lat_coord for j in lon_coord]
    all_coords = np.array(all_coords).T
    return all_coords

def parse_grb_file(file_path: str, area_idx, features_to_use):
    
    grb = pygrib.open(file_path)
    data = grb.read( )
    if not data:
        print(f'data file {file_path} is empty, skipping the file...')
        return {}
    
    lat, lon = grb.message(1).latlons() 
    lat = lat[:, 0]
    lon = lon[0, :]
    
    lat = lat[area_idx[0]].flatten().tolist() 
    lon = lon[area_idx[1]].flatten().tolist()
    
    num_areas = len(lat) 
    curr_valid_time = None
    
    cleaned_data = {'valid_date': [], 'lat': [], 'lon': []}
    
    valid_date_mismatch = 0 
    for attr in data:
        if features_to_use is not None and attr.name not in features_to_use:
            continue
        valid_date = attr.validDate.strftime('%Y-%m-%d %H:%M:%S')
        attr_value = attr.values[area_idx[0], area_idx[1]].flatten( ).tolist()
        
        if attr.name not in cleaned_data.keys():
            cleaned_data[attr.name] = []
        if curr_valid_time != valid_date:
            valid_dt = [valid_date] * num_areas
            cleaned_data['valid_date'].extend(valid_dt) 
            cleaned_data['lat'].extend(lat)
            cleaned_data['lon'].extend(lon)
            
            curr_valid_time = valid_date
            
        cleaned_data[attr.name].extend(attr_value)
        
    return cleaned_data

#############
# 100m u&v

file_path = '100m_u_v.grib'
coords = get_full_grids((0, 64, 0, 44))
features_to_use = ['100 metre U wind component', '100 metre V wind component']

file_path1 = str(file_path).encode('utf-8')
data = parse_grb_file(file_path1, coords, features_to_use) 

reanalysis_df_100_u_v = pd.DataFrame(data)
del data
reanalysis_df_100_u_v['valid_date'] = pd.to_datetime(reanalysis_df_100_u_v['valid_date']) + pd.Timedelta(8, unit='hour')
nwp_data1 = reanalysis_df_100_u_v.drop_duplicates(subset=['valid_date', 'lon', 'lat'], keep='first')
del reanalysis_df_100_u_v

nwp_data1 = nwp_data1.sort_values(by=['valid_date', 'lat', 'lon'],ascending=[True, False, True])