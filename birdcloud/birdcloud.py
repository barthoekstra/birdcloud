"""
description: Bird point cloud builder
author: Bart Hoekstra
"""
import math

import h5py
import numpy as np
import time
import wradlib
import pandas as pd
from datetime import datetime


class BirdCloud:

    def __init__(self, range_limit):
        self.source = None
        self.radar = dict()
        self.product = dict()
        self.pointcloud = pd.DataFrame()
        self.range_limit = range_limit
        self.projection = None

    def from_raw_knmi_file(self, filepath):
        start_time = time.time()

        f = h5py.File(filepath, 'r')

        self.parse_knmi_metadata(f)
        self.extract_knmi_scans(f)

        print('Elapsed time: {}'.format(time.time() - start_time))

    def parse_knmi_metadata(self, file):
        # Radar
        self.radar['name'] = file['radar1'].attrs.get('radar_name').decode('UTF-8')
        latlon = file['radar1'].attrs.get('radar_location')
        self.radar['latitude'] = latlon[1]
        self.radar['longitude'] = latlon[0]
        self.radar['altitude'] = self.radar_metadata[self.radar['name']]['altitude']
        self.radar['polarization'] = self.radar_metadata[self.radar['name']]['polarization']

        # Radar product
        dt_format = '%d-%b-%Y;%H:%M:%S.%f'
        dt_start = file['overview'].attrs.get('product_datetime_start').decode('UTF-8')
        dt_end = file['overview'].attrs.get('product_datetime_end').decode('UTF-8')
        self.product['datetime_start'] = datetime.strptime(dt_start, dt_format)
        self.product['datetime_end'] = datetime.strptime(dt_end, dt_format)

    def extract_knmi_scans(self, file):
        for group in file:
            if file[group].name in self.available_scans:
                scan = dict()

                scan['elevation_angle'] = file[group].attrs.get('scan_elevation')[0]
                scan['n_range_bins'] = file[group].attrs.get('scan_number_range')[0]
                scan['n_azim_bins'] = file[group].attrs.get('scan_number_azim')[0]
                scan['bin_range'] = file[group].attrs.get('scan_range_bin')[0]
                scan['n_range_bins_limit'] = self.calculate_bin_range_limit(self.range_limit, scan['bin_range'],
                                                                            scan['n_range_bins'])
                site_coords = [self.radar['longitude'], self.radar['latitude'], self.radar['altitude']]

                scan['x'], scan['y'], scan['z'] = self.calculate_xyz(scan['elevation_angle'],
                                                                     scan['n_range_bins_limit'],
                                                                     scan['n_azim_bins'], scan['bin_range'],
                                                                     site_coords, range_min=0)

                for dataset in file[group]:
                    if dataset == 'calibration':
                        continue

                    calibration_path = '/{}/calibration/'.format(group)
                    calibration_identifier = dataset.replace('scan', 'calibration').replace('data', 'formulas')
                    calibration_formula = file[calibration_path].attrs.get(calibration_identifier)

                    dataset_path = '/{}/{}'.format(group, dataset)
                    dataset_name = dataset.lstrip('scan_').rstrip('_data')

                    if dataset_name in self.excluded_datasets['DualPol']:
                        continue

                    raw_data = file[dataset_path].value[:, 0:scan['n_range_bins_limit']]

                    if calibration_formula is not None:
                        formula = str(calibration_formula).split('*PV')
                        gain = float(formula[0][7:])
                        offset = float(formula[1][1:-2])

                        corrected_data = raw_data * gain + offset  # @TODO: Also converts 0 values with offset, is this correct?
                        scan[dataset_name] = corrected_data.flatten()

                    else:
                        raw_data = np.tile(raw_data, (scan['n_range_bins'], 1))
                        raw_data = np.transpose(raw_data)
                        scan[dataset_name] = raw_data.flatten()  # @TODO: This data is multidimensional, so needs to be repeated for all distances

                df_scan = pd.DataFrame.from_dict(scan, orient='columns')

                self.pointcloud = self.pointcloud.append(df_scan)

    def calculate_bin_range_limit(self, distance_limit, bin_range, n_range_bins):
        bins = math.floor(distance_limit / bin_range)
        if bins > n_range_bins:
            return n_range_bins
        else:
            return bins

    def calculate_xyz(self, elevation_angle, n_range_bins, n_azim_bins, bin_range, sitecoords=None, range_min=None):
        if range_min is None:
            range_min = 0

        if sitecoords is None:
            sitecoords = (0, 0)

        range_max = range_min + n_range_bins * bin_range
        ranges = np.linspace(range_min, range_max, n_range_bins)
        azimuths = np.arange(0, n_azim_bins)

        polargrid = np.meshgrid(ranges, azimuths)

        xyz, self.projection = wradlib.georef.polar.spherical_to_xyz(polargrid[0], polargrid[1], elevation_angle,
                                                                     sitecoords)

        xyz = xyz.flatten().reshape(n_azim_bins * n_range_bins, 3)

        return xyz[:, 0], xyz[:, 1], xyz[:, 2]

    def to_csv(self, file_path):
        self.pointcloud.to_csv(file_path, index=False)

    radar_metadata = {
        'DeBilt': {'altitude': 44, 'polarization': 'SinglePol'},
        'Den Helder': {'altitude': 51, 'polarization': 'DualPol'},
        'Herwijnen': {'altitude': 27.7, 'polarization': 'DualPol'}
    }

    available_scans = {'/scan1', '/scan2', '/scan3', '/scan4', '/scan5', '/scan6', '/scan7', '/scan8',
                       '/scan9', '/scan10', '/scan11', '/scan12', '/scan13', '/scan14', '/scan15', '/scan16'}

    available_datasets = {
        'SinglePol': {
            'uZ': 'Uncorrected reflectivity',
            'V': 'Radial velocity',
            'Z': 'Reflectivity (corrected)',
            'W': 'Spectral width of radial velocity',
            'TX_power': 'Total reflectivity factor'
        },
        'DualPol': {
            'CCOR': 'Clutter correction (horizontally polarized)',
            'CCORv': 'Clutter correction (vertically polarized)',
            'CPA': 'Clutter phase alignment (horizontally polarized)',
            'CPAv': 'Clutter phase alignment (vertically polarized)',
            'KDP': 'Specific differential phase',
            'PhiDP': 'Differential phase',
            'RhoHV': 'Correlation between Z(h) and Zv',
            'SQI': 'Signal quality index (horizontally polarized)',
            'SQIv': 'Signal quality index (vertically polarized)',
            'TX_power': 'Total reflectivity factor',
            'uPhiDP': 'Unsmoothed differential phase',
            'uZ': 'Uncorrected reflectivity (horizontally polarized)',
            'uZv': 'Uncorrected reflectivity (vertically polarized)',
            'V': 'Radial velocity (horizontally polarized)',
            'Vv': 'Radial velocity (vertically polarized)',
            'W': 'Spectral width of radial velocity (horizontally polarized)',
            'Wv': 'Spectral width of radial velocity (vertically polarized)',
            'Z': 'Reflectivity (corrected, horizontally polarized)',
            'Zv': 'Reflectivity (corrected, vertically polarized)'
        }
    }

    excluded_datasets = {
        'SinglePol': {'CCOR', 'CCORv', 'CPA', 'CPAv', 'SQI', 'SQIv', 'TX_power'},
        'DualPol': {'CCOR', 'CCORv', 'CPA', 'CPAv', 'SQI', 'SQIv', 'TX_power'},
    }


if __name__ == '__main__':
    b = BirdCloud(50)
    # b.from_raw_knmi_file('../data/raw/RAD_NL60_VOL_NA_201610020500.h5')
    b.from_raw_knmi_file('../data/raw/RAD_NL62_VOL_NA_201802010000.h5')
    # b.to_csv('../data/processed/RAD_NL60_VOL_NA_201610020500.csv')
    # b.to_csv('../data/processed/RAD_NL62_VOL_NA_201802010000.csv')
