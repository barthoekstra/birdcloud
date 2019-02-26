"""
description: Converts KNMI formatted HDF5 files to ODIM formatted HDF5 files
author: Bart Hoekstra

@TODO: Add logging so conversion errors can be stored
"""
import glob
import subprocess
import time

knmi_files = glob.glob('../../data/raw/*.h5')

start_time = time.time()

for knmi_file in knmi_files:
    print('Converting: {}'.format(knmi_file))

    odim_file = knmi_file.replace('.h5', '_ODIM.h5').replace('../data/raw/', '../data/interim/')

    cmd = './KNMI_vol_h5_to_ODIM_h5 {} {}'.format(odim_file, knmi_file)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        continue

print('Elapsed time: {}'.format(time.time() - start_time))