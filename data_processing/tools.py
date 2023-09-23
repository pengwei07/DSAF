from typing import Callable, Dict, List, Union
import numpy as np

'''
def check_input_dict(func: Callable):
    """
    check input_dict for two situations:
    1. empty
    2. wrong format

    In both situations, prevent the file from merging.

    Args:
        func: parse_grb_file

    Returns:

    """

    def inner(*args):
        dict_in = func(*args)
        if not dict_in:
            return dict_in, False

        length = len(dict_in['valid_date'])
        combine = True

        for k, v in dict_in.items():
            if length != len(v):
                print(f"file {dict_in['valid_date'][0]}, {k} has some problem, skipping the file ...")
                combine = False
                break

        return dict_in, combine

    return inner


def get_area_idx(area, unique_lats, unique_lons):
    """
    Get the numerial area index acoording to the position of the selected area in the given grids.

    Args:
        area: tuple, (lat_max, lat_min, lon_min, lon_max)
        unique_lats: list, [N, S], descending order;
        unique_lons: list, [W, E], ascending order

    Returns:
        area_idx: tuple
    """
    north, south, west, east = np.round(area, 1)

    ## First, test if the selected area is a sub area of the boundary:

    if (north < south) or (west > east) or (north > unique_lats[0]) or (south < unique_lats[-1]) or (
            west < unique_lons[0]) or (
            east > unique_lons[-1]):
        print('Wrong Area Selection! Please select another one.')
        return

    selected_lats_upper = (np.around((unique_lats[0] - north) * 10, 1)).astype(int)
    selected_lats_lower = (np.around((unique_lats[0] - south) * 10, 1)).astype(int)
    selected_lons_right = (np.around((east - unique_lons[0]) * 10, 1)).astype(int)
    selected_lons_left = (np.around((west - unique_lons[0]) * 10, 1)).astype(int)
    area_idx = (selected_lats_upper, selected_lats_lower, selected_lons_left, selected_lons_right)

    return area_idx


def merge_dict(dict_1: Dict, dict_2: Dict):
    """
    Merge dict_2 into dict_1.

    Args:
        dict_1: the reference dict.
        dict_2: the target dict.

    Returns:

    """
    for k, v in dict_2.items():
        dict_1[k].extend(v)

    return dict_1
'''

def get_full_grids(area_idx):
    """
    Return the full coordinate grids given area. All boundaries included.
    Args:
        area_idx: tuple.

    Returns:

    """
    # N, S, W, E = eval(area_idx[0]), eval(area_idx[1]), eval(area_idx[2]), eval(area_idx[3])
    N, S, W, E = area_idx
    lat_coord = np.arange(N, (S + 1))
    lon_coord = np.arange(W, (E + 1))
    all_coords = [[i, j] for i in lat_coord for j in lon_coord]
    all_coords = np.array(all_coords).T
    return all_coords
