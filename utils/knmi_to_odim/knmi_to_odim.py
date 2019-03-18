"""
description: Converts KNMI formatted HDF5 files to ODIM formatted HDF5 files
author: Bart Hoekstra

@TODO: Add logging so conversion errors can be stored
"""
import os
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
import tarfile

path_raw = Path('/') / 'Users' / 'barthoekstra' / 'Development' / 'birdcloud' / 'data' / 'raw' / '201708'
print(path_raw)
path_converted = path_raw.parent.parent / 'interim' / '201708_ODIM'

start_time = time.time()


def unpack_file(file):
    tar = tarfile.open(file)
    tar.extractall(path=path_raw)
    tar.close()


with Pool(processes=6) as pool:
    files = [file for file in path_raw.glob('**/*.tar')]
    pool.map(unpack_file, files)


# Apparently unpacking is finished

def convert_knmi_file(file):
    odim_file = path_converted / file.name.replace('.h5', '_ODIM.h5')

    cmd = './KNMI_vol_h5_to_ODIM_h5 {} {}'.format(odim_file, file)

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
    else:
        os.remove(file)

with Pool(processes=6) as pool:
    files = [file for file in path_raw.glob('*.h5')]
    pool.map(convert_knmi_file, files)


print('Elapsed time: {}'.format(time.time() - start_time))