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

    def __init__(self):
        """
        Prepares BirdCloud object

        The object contains radar, product and projection metadata and a potentially range limited point cloud.
        """
        self.source = None
        self.radar = dict()
        self.product = dict()
        self.pointcloud = pd.DataFrame()
        self.range_limit = None
        self.projection = None

    def from_raw_knmi_file(self, filepath, range_limit=None):
        """
        Builds point cloud from KNMI radar HDF5 file

        :param filepath: path to the raw KNMI radar HDF5 file
        :param range_limit: None or iterable containing both minimum and maximum range of the point cloud from the radar
            site.
        """
        f = h5py.File(filepath, 'r')
        self.source = filepath
        self.range_limit = range_limit

        self.parse_knmi_metadata(f)
        self.extract_knmi_scans(f)

    def parse_knmi_metadata(self, file):
        """
        Parses KNMI metadata from HDF5 file about the radar itself and the provided radar products
        :param file: HDF5 file object
        """
        self.radar['name'] = file['radar1'].attrs.get('radar_name').decode('UTF-8')
        latlon = file['radar1'].attrs.get('radar_location')
        self.radar['latitude'] = latlon[1]
        self.radar['longitude'] = latlon[0]
        self.radar['altitude'] = self.radar_metadata[self.radar['name']]['altitude']
        self.radar['polarization'] = self.radar_metadata[self.radar['name']]['polarization']

        dt_format = '%d-%b-%Y;%H:%M:%S.%f'
        dt_start = file['overview'].attrs.get('product_datetime_start').decode('UTF-8')
        dt_end = file['overview'].attrs.get('product_datetime_end').decode('UTF-8')
        self.product['datetime_start'] = datetime.strptime(dt_start, dt_format)
        self.product['datetime_end'] = datetime.strptime(dt_end, dt_format)

    def extract_knmi_scans(self, file):
        """
        Iterates over all scans and corresponding datasets in the KNMI HDF5 raw radar files and builds the point cloud.
        Additionally, all missing data values are removed and differential reflectivity (ZDR) is calculated.

        @TODO: Consider different implementations which do not require constant concatenation of new scans
        :param file: HDF5 file object
        """
        for group in file:
            if file[group].name in self.available_scans:

                if file[group].name in self.excluded_scans:
                    continue

                scan = dict()
                z_offset = None

                scan['elevation_angle'] = file[group].attrs.get('scan_elevation')[0]
                n_range_bins = file[group].attrs.get('scan_number_range')[0]
                n_azim_bins = file[group].attrs.get('scan_number_azim')[0]
                bin_range = file[group].attrs.get('scan_range_bin')[0]
                site_coords = [self.radar['longitude'], self.radar['latitude'], self.radar['altitude'] / 1000]

                bin_range_min, bin_range_max = self.calculate_bin_range_limits(self.range_limit, bin_range,
                                                                               n_range_bins)

                scan['x'], scan['y'], scan['z'], scan['r'], scan['phi'] = self.calculate_xyz(site_coords,
                                                                                             scan['elevation_angle'],
                                                                                             n_azim_bins, bin_range,
                                                                                             bin_range_min,
                                                                                             bin_range_max)

                for dataset in file[group]:
                    if dataset == 'calibration':
                        continue

                    calibration_path = '/{}/calibration/'.format(group)
                    calibration_identifier = dataset.replace('scan', 'calibration').replace('data', 'formulas')
                    calibration_formula = file[calibration_path].attrs.get(calibration_identifier)
                    dataset_path = '/{}/{}'.format(group, dataset)
                    dataset_name = dataset.lstrip('scan_').rstrip('_data')

                    if dataset_name in self.excluded_datasets[self.radar['polarization']]:
                        continue

                    raw_data = file[dataset_path].value[:, bin_range_min:bin_range_max]

                    if calibration_formula is not None:
                        calibration_formula = calibration_formula.decode('UTF-8')
                        formula = str(calibration_formula).split('*PV')
                        gain = float(formula[0][4:])
                        offset = float(formula[1][1:])

                        if dataset_name == 'Z':
                            z_offset = offset

                        corrected_data = raw_data * gain + offset
                        scan[dataset_name] = corrected_data.flatten()
                    else:
                        raw_data = np.tile(raw_data, (n_range_bins, 1))
                        raw_data = np.transpose(raw_data)
                        scan[dataset_name] = raw_data.flatten()

                df_scan = pd.DataFrame.from_dict(scan, orient='columns')

                """The missing data value is 0 (see calibration -> calibration_missing_data), but this gets converted
                to a different value using the offset. Since we don't need empty points in the point cloud, we remove
                all the records where the Z value is equal to the Z offset."""
                df_scan = df_scan[df_scan['Z'] != z_offset]

                df_scan['ZDR'] = df_scan['Z'] - df_scan['Zv']

                self.pointcloud = self.pointcloud.append(df_scan)

    def calculate_bin_range_limits(self, range_limit, bin_range, n_range_bins):
        """
        Calculates range of bins to select for all to fall within provided range_limit.

        If the lower limit is None, it will be converted to 0. If the upper limit is None, it will be converted to the
        maximum value of the range, i.e. n_range_bins.

        E.g. this function will return the 6th bin as bins_min if the minimum range limit is set to 5 (km) and the
            bin_range is 0.900: 5/0.900 = 5.5555 -> 6

        :param range_limit: iterable containing successively a minimum and maximum range
        :param bin_range: range covered by a single bin in the same units as range_limit
        :param n_range_bins: the number of range bins within a scan
        :return: indexes for the first (bins_min) and last (bins_max) bins that fall within the given range_limit
        """
        if range_limit is None:
            range_limit = [None, None]

        minimum = range_limit[0] if range_limit[0] is not None else 0
        maximum = range_limit[1] if range_limit[1] is not None else n_range_bins

        bins_min = math.ceil(minimum / bin_range)
        if bins_min > n_range_bins:
            raise ValueError('Minimum range set too high: no datapoints remaining.')

        bins_max = math.floor(maximum / bin_range)
        if bins_max > n_range_bins:
            bins_max = n_range_bins

        return bins_min, bins_max

    def calculate_xyz(self, sitecoords, elevation_angle, n_azim_bins, bin_range, bin_range_min, bin_range_max):
        """
        Calculates X, Y and Z coordinates for centers of all radar bins using wradlib and sets self.projection to the
        corresponding georeferencing information.

        :param sitecoords: iterable containing coordinates and altitude of radar site, in the order of: longitude,
            latitude, altitude
        :param elevation_angle: elevation angle of the radar scan
        :param n_azim_bins: number of azimuthal bins of the radar scan (usually 360)
        :param bin_range: range covered by every bin (usually in kilometers). Should be in the same units as the radar
            altitude
        :param bin_range_min: index of the first range bin to calculate X, Y and Z coordinates for
        :param bin_range_max: index of the last range bin to calculate X, Y and Z coordinates for
        :return: numpy arrays for the X, Y and Z coordinates
        """
        if sitecoords is None:
            sitecoords = (0, 0)

        n_range_bins = bin_range_max - bin_range_min
        range_min = bin_range_min * bin_range
        range_max = bin_range_max * bin_range
        ranges = np.linspace(range_min, range_max, n_range_bins)
        azimuths = np.arange(0, n_azim_bins)

        polargrid = np.meshgrid(ranges, azimuths)
        r = polargrid[0].flatten()
        phi = polargrid[1].flatten()

        xyz, self.projection = wradlib.georef.polar.spherical_to_xyz(polargrid[0], polargrid[1], elevation_angle,
                                                                     sitecoords)

        xyz = xyz.flatten().reshape(n_azim_bins * n_range_bins, 3)

        return xyz[:, 0], xyz[:, 1], xyz[:, 2], r, phi

    def to_csv(self, file_path):
        """
        Exports the point cloud to a CSV file.
        :param file_path: path to CSV file. If the file does not exist yet, it will be created.
        """
        self.pointcloud.to_csv(file_path, index=False)

    radar_metadata = {
        'DeBilt': {'altitude': 44, 'polarization': 'SinglePol'},
        'Den Helder': {'altitude': 51, 'polarization': 'DualPol'},
        'Herwijnen': {'altitude': 27.7, 'polarization': 'DualPol'}
    }

    available_scans = {'/scan1', '/scan2', '/scan3', '/scan4', '/scan5', '/scan6', '/scan7', '/scan8',
                       '/scan9', '/scan10', '/scan11', '/scan12', '/scan13', '/scan14', '/scan15', '/scan16'}

    excluded_scans = {'/scan1', '/scan7', '/scan16'}

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
    start_time = time.time()
    b = BirdCloud()
    b.from_raw_knmi_file('../data/raw/RAD_NL62_VOL_NA_201801010025.h5', [0, 50])
    b.to_csv('../data/processed/RAD_NL62_VOL_NA_201801010025.csv')
    print('Elapsed time: {}'.format(time.time() - start_time))
