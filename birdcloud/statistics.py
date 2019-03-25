"""
description: Bird point cloud statistics
author: Bart Hoekstra
"""

import math
import time

from birdcloud import BirdCloud
import bottleneck as bn


def calculate_texture(bc, point_elevation, point_azimuth, point_range, products, bins=None):
    """

    :param bc:
    :param point_elevation:
    :param point_azimuth:
    :param point_range:
    :param products: List of products to calculate texture for
    :param bins: Iterable containing numbers of bins to include in texture window surrounding a centerpoint.
        Should be e.g. a list or tuple in order [nr_elevations, nr_azimuthal_bins, nr_range_bins].
    :return:
    """
    if not isinstance(bc, BirdCloud):
        raise TypeError('bc must be a BirdCloud object')

    if products is None:
        raise ValueError('products has to be set to calculate texture')

    if bins is None:
        bins = [1, 1, 1]

    elevations = get_surrounding_elevations(bc, point_elevation, bins[0])
    ground_distance_to_radar = calculate_ground_distance_to_radar(point_elevation, point_range)
    equidistant_ranges = calculate_equidistant_range(elevations, ground_distance_to_radar)

    texture = {product: [] for product in products}

    for index, elevation in enumerate(elevations):
        window = get_scan_window_polar(bc, elevation, point_azimuth, equidistant_ranges[index], bins)

        for product in texture.keys():
            product_flat = window[product].stack(a=('azimuth', 'range')).values.tolist()
            texture[product].extend(product_flat)

    return {product: bn.nanstd(texture[product]) for product in products}

def calculate_texture_dask(bc, row, products, bins=None):
    print(row)


def get_scan_window_polar(bc, point_elevation, point_azimuth, point_range, bins):
    elevation_bins, azim_bins, range_bins = bins

    scan = bc.scans[str(point_elevation)]

    azimuths = range(point_azimuth - azim_bins, point_azimuth + azim_bins + 1, 1)
    azimuth_range = [azimuth % 360 for azimuth in azimuths]

    bin_range = scan.attrs['bin_range']
    range_range = slice(point_range - bin_range * range_bins - bin_range * 0.5,
                        point_range + bin_range * range_bins + bin_range * 0.5)

    return scan.sel(azimuth=azimuth_range, range=range_range)


def calculate_ground_distance_to_radar(point_elevation, point_range):
    return point_range * math.cos(math.radians(point_elevation))


def calculate_equidistant_range(elevations, ground_distance_to_radar):
    return [ground_distance_to_radar / math.cos(math.radians(elevation)) for elevation in elevations]


def get_surrounding_elevations(bc, point_elevation, elevation_bins):
    """
    Returns surrounding elevations of given point_elevation value if they exist. If point_elevation is either the
    highest or lowest elevation, no higher or lower elevations respectively are returned.

    :param bc: BirdCloud object
    :param point_elevation: Float value of current elevation
    :param elevation_bins: Integer value of number of elevations to return surrounding the current elevation
    :return: List of surrounding elevations
    """
    current_elevation_index = bc.elevations.index(point_elevation)

    lowest_elevation = current_elevation_index - elevation_bins
    lowest_elevation = 0 if lowest_elevation < 0 else lowest_elevation

    return bc.elevations[lowest_elevation:current_elevation_index + elevation_bins + 1]


if __name__ == '__main__':
    start_time = time.time()

    bc = BirdCloud()
    bc.from_odim_file('../data/test/RAD_NL61_VOL_NA_2330_ODIM.h5', [5, 25])

    i = 0
    rows = len(bc.pointcloud.index)
    texture_column = {'DBZH': [], 'VRADH': [], 'ZDR': []}

    for index, row in bc.pointcloud.iterrows():
        # if i == 10:
        #     break

        texture = calculate_texture(bc, float(index[0]), index[1], index[2], ['DBZH', 'VRADH', 'ZDR'], bins=[1, 1, 1])

        for product_texture, texture in texture.items():
            texture_column[product_texture].extend([texture])

        if i % 1000 == 0:
            print('{}/{}'.format(i, rows))

        i += 1

    bc.pointcloud['DBZHtex'] = texture_column['DBZH']
    bc.pointcloud['VRADHtex'] = texture_column['VRADH']
    bc.pointcloud['ZDRtex'] = texture_column['ZDR']
    bc.to_csv('../data/test/RAD_NL61_VOL_NA_2330_ODIM_tex.csv')

    print('Elapsed time: {}'.format(time.time() - start_time))



