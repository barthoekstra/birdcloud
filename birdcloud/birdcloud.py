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
        self.calculate_additional_metrics()
        self.drop_na_rows()
        self.set_column_order()

    def from_odim_file(self, filepath, range_limit=None):
        """
        Builds point cloud from ODIM formatted polar volume

        :param filepath: path to the ODIM formatted HDF5 file
        :param range_limit: None or iterable containing both minimum and maximum range of the point cloud from the radar
            site.
        """
        f = h5py.File(filepath, 'r')
        self.source = filepath
        self.range_limit = range_limit

        self.parse_odim_metadata(f)
        self.extract_odim_scans(f)
        self.calculate_additional_metrics()
        self.drop_na_rows()
        self.set_column_order()

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

    def parse_odim_metadata(self, file):
        """
        Parses ODIM metadata from HDF5 file about radar itself and the provided radar products

        Volume scan start and end times are derived from start time of the 1st scan and end time of the 16th scan, as
        scans are numbered chronologically.

        Note: Fetched values are converted to certain types to ensure consistency between KNMI and ODIM formatted radar
            volume files. @TODO: Remove if unnecessary. (Probably is).
        :param file: HDF5 file object
        """
        source = dict(pair.split(':') for pair in file['what'].attrs.get('source').decode('UTF-8').split(','))
        self.radar['name'] = source['PLC'] if source.get('PLC') is not None else source.get('WMO')
        self.radar['latitude'] = file['where'].attrs.get('lat')[0]
        self.radar['longitude'] = file['where'].attrs.get('lon')[0]
        self.radar['altitude'] = file['where'].attrs.get('height')[0]
        self.radar['polarization'] = self.radar_metadata[self.radar['name']]['polarization']

        time_start = datetime.strptime(file['dataset1']['what'].attrs.get('starttime').decode('UTF-8'), '%H%M%S').time()
        date_start = datetime.strptime(file['dataset1']['what'].attrs.get('startdate').decode('UTF-8'), '%Y%m%d').date()
        self.product['datetime_start'] = datetime.combine(date_start, time_start)
        time_end = datetime.strptime(file['dataset16']['what'].attrs.get('endtime').decode('UTF-8'), '%H%M%S').time()
        date_end = datetime.strptime(file['dataset16']['what'].attrs.get('enddate').decode('UTF-8'), '%Y%m%d').date()
        self.product['datetime_end'] = datetime.combine(date_end, time_end)

    def extract_knmi_scans(self, file):
        """
        Iterates over all scans and corresponding datasets in the KNMI HDF5 raw radar files and builds the point cloud.
        Additionally, all missing data values are removed and differential reflectivity (ZDR) is calculated.

        @TODO: Consider different implementations which do not require constant concatenation of new scans
        :param file: KNMI HDF5 file object
        """
        for group in file:
            if file[group].name.startswith('scan', 1):

                if file[group].name in self.excluded_scans:
                    continue

                scan = dict()

                scan['elev'] = file[group].attrs.get('scan_elevation')[0]
                n_range_bins = file[group].attrs.get('scan_number_range')[0]
                n_azim_bins = file[group].attrs.get('scan_number_azim')[0]
                bin_range = file[group].attrs.get('scan_range_bin')[0]
                site_coords = [self.radar['longitude'], self.radar['latitude'], self.radar['altitude'] / 1000]

                bin_range_min, bin_range_max = self.calculate_bin_range_limits(self.range_limit, bin_range,
                                                                               n_range_bins)

                scan['x'], scan['y'], scan['z'], scan['r'], scan['phi'] = self.calculate_xyz(site_coords,
                                                                                             scan['elev'],
                                                                                             n_azim_bins, bin_range,
                                                                                             bin_range_min,
                                                                                             bin_range_max)

                for dataset in file[group]:
                    if not dataset.startswith('scan_'):
                        continue

                    quantity = dataset.lstrip('scan_').rstrip('_data')

                    if quantity in self.excluded_datasets[self.radar['polarization']]:
                        continue

                    calibration_identifier = 'calibration_{}_formulas'.format(quantity)
                    calibration_formula = file[group]['calibration'].attrs.get(calibration_identifier).decode('UTF-8')
                    gain, offset = calibration_formula.lstrip('GEO=').split('*PV+')
                    gain = np.float64(gain)
                    offset = np.float64(offset)
                    nodata = file[group]['calibration'].attrs.get('calibration_missing_data')
                    undetect = nodata

                    raw_data = file[group][dataset].value[:, bin_range_min:bin_range_max]
                    missing = np.logical_or(raw_data == nodata, raw_data == undetect)

                    corrected_data = raw_data * gain + offset
                    corrected_data[missing] = np.nan

                    odim_quantity = self.available_datasets[self.radar['polarization']][quantity]['ODIM']

                    scan[odim_quantity] = corrected_data.flatten()

                df_scan = pd.DataFrame.from_dict(scan, orient='columns')

                self.pointcloud = self.pointcloud.append(df_scan)

    def extract_odim_scans(self, file):
        """
        Iterates over all scans and corresponding datasets in the ODIM formatted HDF5 files and builds the point cloud.
        Additionally, all missing data values are removed and differential reflectivity (ZDR) is calculated.

        Note: Units used in KNMIs raw file and ODIM files are different, so we convert meters to kilometers.

        :param file: ODIM HDF5 file object
        """
        for group in file:
            if file[group].name.startswith('dataset', 1):

                if file[group].name in self.excluded_scans:
                    continue

                scan = dict()

                scan['elev'] = file[group]['where'].attrs.get('elangle')[0]
                n_range_bins = file[group]['where'].attrs.get('nbins')[0]
                n_azim_bins = file[group]['where'].attrs.get('nrays')[0]
                bin_range = file[group]['where'].attrs.get('rscale')[0] / 1000

                site_coords = [self.radar['longitude'], self.radar['latitude'], self.radar['altitude'] / 1000]

                bin_range_min, bin_range_max = self.calculate_bin_range_limits(self.range_limit, bin_range,
                                                                               n_range_bins)

                scan['x'], scan['y'], scan['z'], scan['r'], scan['phi'] = self.calculate_xyz(site_coords,
                                                                                             scan['elev'],
                                                                                             n_azim_bins, bin_range,
                                                                                             bin_range_min,
                                                                                             bin_range_max)

                for dataset in file[group]:
                    if not dataset.startswith('data'):
                        continue

                    quantity = file[group][dataset]['what'].attrs.get('quantity').decode('UTF-8')

                    if quantity in self.excluded_datasets[self.radar['polarization']]:
                        continue

                    gain = file[group][dataset]['what'].attrs.get('gain')[0]
                    offset = file[group][dataset]['what'].attrs.get('offset')[0]
                    nodata = file[group][dataset]['what'].attrs.get('nodata')[0]
                    undetect = file[group][dataset]['what'].attrs.get('undetect')[0]

                    raw_data = file[group][dataset]['data'].value[:, bin_range_min:bin_range_max]
                    missing = np.logical_or(raw_data == nodata, raw_data == undetect)

                    corrected_data = raw_data * gain + offset
                    corrected_data[missing] = np.nan

                    scan[quantity] = corrected_data.flatten()

                df_scan = pd.DataFrame.from_dict(scan, orient='columns')

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

    def calculate_additional_metrics(self):
        """
        Triggers calculation of other metrics, such as ZDR calculation, textures etc.
        """
        self.calculate_differential_reflectivity()

    def calculate_differential_reflectivity(self):
        """
        Calculates differential reflectivity or ZDR, defined as DBZH - DBZV (following Stepanian et al., 2016).
        """
        self.pointcloud['ZDR'] = self.pointcloud['DBZH'] - self.pointcloud['DBZV']

    def drop_na_rows(self, subset=None):
        """
        Drops rows from the file where columns in subset containg NA/NaN values. By default this is done for rows that
        contain no values for DBZH and VRADH.

        :param subset: Iterable of column names that cannot have NA/NaN values in rows.
        """
        if subset is None:
            subset = ['DBZH', 'VRADH']

        self.pointcloud.dropna(subset=subset, inplace=True)

    def set_column_order(self):
        """
        Orders columns in order defined in self.column_order for both single-pol and dual-pol polarizations.
        """
        order = self.column_order[self.radar['polarization']]
        columns_unordered = list(self.pointcloud.columns)
        columns_ordered = [variable for variable in order if variable in columns_unordered]
        self.pointcloud = self.pointcloud[columns_ordered]

    def to_csv(self, file_path):
        """
        Exports the point cloud to a CSV file.
        :param file_path: path to CSV file. If the file does not exist yet, it will be created.
        """
        self.pointcloud.to_csv(file_path, na_rep="NaN", quotechar='"', index=False)

    radar_metadata = {
        'DeBilt': {'altitude': 44, 'polarization': 'SinglePol'},
        'Den Helder': {'altitude': 51, 'polarization': 'DualPol'},
        'Herwijnen': {'altitude': 27.7, 'polarization': 'DualPol'}
    }

    excluded_scans = {'/scan1', '/scan7', '/scan16', '/dataset1', '/dataset7', '/dataset16'}

    available_datasets = {
        'SinglePol': {
            'uZ': {'description': 'Uncorrected reflectivity', 'ODIM': 'TH'},
            'V': {'description': 'Radial velocity', 'ODIM': 'VRADH'},
            'Z': {'description': 'Reflectivity (corrected)', 'ODIM': 'DBZH'},
            'W': {'description': 'Spectral width of radial velocity', 'ODIM': 'WRADH'},
            'TX_power': {'description': 'Total reflectivity factor', 'ODIM': None}
        },
        'DualPol': {
            'CCOR': {'description': 'Clutter correction (horizontally polarized)', 'ODIM': 'CCORH'},
            'CCORv': {'description': 'Clutter correction (vertically polarized)', 'ODIM': 'CCORV'},
            'CPA': {'description': 'Clutter phase alignment (horizontally polarized)', 'ODIM': 'CPAH'},
            'CPAv': {'description': 'Clutter phase alignment (vertically polarized)', 'ODIM': 'CPAV'},
            'KDP': {'description': 'Specific differential phase', 'ODIM': 'KDP'},
            'PhiDP': {'description': 'Differential phase', 'ODIM': 'PHIDP'},
            'RhoHV': {'description': 'Correlation between Z(h) and Zv', 'ODIM': 'RHOHV'},
            'SQI': {'description': 'Signal quality index (horizontally polarized)', 'ODIM': 'SQIH'},
            'SQIv': {'description': 'Signal quality index (vertically polarized)', 'ODIM': 'SQIV'},
            'TX_power': {'description': 'Total reflectivity factor', 'ODIM': None},
            'uPhiDP': {'description': 'Unsmoothed differential phase', 'ODIM': 'PHIDPU'},
            'uZ': {'description': 'Uncorrected reflectivity (horizontally polarized)', 'ODIM': 'TH'},
            'uZv': {'description': 'Uncorrected reflectivity (vertically polarized)', 'ODIM': 'TV'},
            'V': {'description': 'Radial velocity (horizontally polarized)', 'ODIM': 'VRADH'},
            'Vv': {'description': 'Radial velocity (vertically polarized)', 'ODIM': 'VRADV'},
            'W': {'description': 'Spectral width of radial velocity (horizontally polarized)', 'ODIM': 'WRADH'},
            'Wv': {'description': 'Spectral width of radial velocity (vertically polarized)', 'ODIM': 'WRADV'},
            'Z': {'description': 'Reflectivity (corrected, horizontally polarized)', 'ODIM': 'DBZH'},
            'Zv': {'description': 'Reflectivity (corrected, vertically polarized)', 'ODIM': 'DBZV'}
        }
    }

    excluded_datasets = {
        'SinglePol': {'CCOR', 'CCORv', 'CPA', 'CPAv', 'SQI', 'SQIv', 'TX_power',  # KNMI HDF5
                      'CCORH', 'CCORV', 'CPAH', 'CPAV', 'SQIH', 'SQIV'},  # ODIM HDF5
        'DualPol': {'CCOR', 'CCORv', 'CPA', 'CPAv', 'SQI', 'SQIv', 'TX_power',  # KNMI HDF5
                    'CCORH', 'CCORV', 'CPAH', 'CPAV', 'SQIH', 'SQIV'}  # ODIM HDF5
    }

    column_order = {
        'SinglePol': ['x', 'y', 'z', 'elev', 'r', 'phi', 'DBZH', 'TH', 'VRADH', 'WRADH', 'TX_power'],
        'DualPol': ['x', 'y', 'z', 'elev', 'r', 'phi', 'DBZH', 'DBZV', 'TH', 'TV', 'VRADH', 'VRADV', 'WRADH', 'WRADV',
                    'PHIDP', 'PHIDPU', 'RHOHV', 'KDP', 'ZDR', 'CCORH', 'CCORV', 'CPAH', 'CPAV', 'SQIH', 'SQIV',
                    'TX_power']
    }


if __name__ == '__main__':
    start_time = time.time()
    b = BirdCloud()
    #b.from_raw_knmi_file('../data/raw/KNMI.h5', [5, 25])
    b.from_odim_file('../data/raw/RAD_NL61_VOL_NA_2005_ODIM.h5', [0, 100])
    b.to_csv('../data/processed/RAD_NL61_VOL_NA_2005_ODIM.csv')
    #b.from_odim_file('../data/raw/deemd_pvol_20170215T0000_10204.h5')
    #b.to_csv('../data/processed/RAD_NL62_VOL_NA_201810282300.csv')
    #b.to_csv('../data/processed/NEXRAD_EXAMPLE.h5')
    print('Elapsed time: {}'.format(time.time() - start_time))