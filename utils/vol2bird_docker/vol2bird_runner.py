"""
description: Processes ODIM formatted radar volume files with vol2bird to generate vertical profiles of birds
author: Bart Hoekstra

vol2Bird generates the following data:
date      - date [UTC]
time      - time [UTC]
HGHT      - height above mean sea level [m]. Alt. bin from HGHT to HGHT+interval)
u         - speed component west to east [m/s]
v         - speed component north to south [m/s]
w         - vertical speed (unreliable!) [m/s]
ff        - horizontal speed [m/s]
dd        - direction [degrees, clockwise from north]
sd_vvp    - VVP radial velocity standard deviation [m/s]
gap       - Angular data gap detected [T/F]
dbz       - Bird reflectivity factor [dBZ]
eta       - Bird reflectivity [cm^2/km^3]
dens      - Bird density [birds/km^3]
DBZH      - Total reflectivity factor (bio+meteo scattering) [dBZ]
n         - number of points VVP bird velocity analysis (u,v,w,ff,dd)
n_dbz     - number of points bird density estimate (dbz,eta,dens)
n_all     - number of points VVP st.dev. estimate (sd_vvp)
n_dbz_all - number of points total reflectivity estimate (DBZH)

# date   time HGHT    u      v       w     ff    dd  sd_vvp gap dbz     eta   dens   DBZH   n   n_dbz n_all n_dbz_all
"""

from pathlib import Path
import time
import docker
from docker import errors
import requests.exceptions
import sqlite3

start_time = time.time()

data_path = Path().cwd().parents[1] / 'data' / 'interim'

# Get Docker client
client = docker.from_env()
container = None

try:
    container = client.containers.get('vol2bird')
    print('The vol2bird Docker container seems to be running already.')
except docker.errors.NotFound:
    volumes = {data_path: {'bind': '/data', 'mode': 'rw'}}
    container = client.containers.run(image='adokter/vol2bird', name='vol2bird', volumes=volumes,
                                      command='sleep infinity', detach=True)
    print('Started the vol2bird Docker container.')
except requests.exceptions.ConnectionError as e:
    print('Docker error: {}\nAre you sure the Docker client is running (e.g. Docker for Mac)?'.format(e))

connection = sqlite3.connect('vertical-profiles.db')
cursor = connection.cursor()

sql_table = ('CREATE TABLE IF NOT EXISTS vertical_profiles ('
             'date integer NOT NULL,'
             'time integer NOT NULL,'
             'height integer NOT NULL,'
             'u real,'
             'v real,'
             'w real,'
             'ff real,'
             'dd real,'
             'sd_vvp real,'
             'gap text,'
             'dbz real,'
             'eta real,'
             'dens real,'
             'DBZH real,'
             'n integer,'
             'n_dbz integer,'
             'n_all integer,'
             'n_dbz_all integer,'
             'PRIMARY KEY (date, time, height)'
             ')')

cursor.execute(sql_table)

sql_vp = ('INSERT INTO vertical_profiles ('
          'date, time, height, u, v, w, ff, dd, sd_vvp, gap, dbz, eta, dens, DBZH, n, n_dbz, n_all, n_dbz_all)'
          'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);')

try:
    container.exec_run('bash -c "cd data"', stdout=True, stderr=True).output.decode('utf-8')
    i = 0
    for file in data_path.glob('*_ODIM.h5'):
        cmd = 'bash -c "cd data && vol2bird {} {}.vp.h5"'.format(file.name, file.name)
        out = container.exec_run(cmd, stdout=True, stderr=True).output.decode('utf-8')
        for line in out.splitlines():
            if line.startswith('#') or line.startswith('Warning: '):
                continue

            data = line.split()
            cursor.execute(sql_vp, data)
            print('Finished file: {}'.format(i))

except docker.errors.APIError as e:
    print('Docker error: {}'.format(e))

connection.commit()

try:
    container.kill()
    container.remove()
    print('Stopped and removed the vol2bird Docker container')
except docker.errors.APIError as e:
    print('Docker error: {}\nDid the Docker client stop running while processing files with vol2bird?'.format(e))

print('Elapsed time: {}'.format(time.time() - start_time))



