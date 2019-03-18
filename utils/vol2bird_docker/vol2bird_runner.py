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
from multiprocessing import Pool, cpu_count, current_process

start_time = time.time()

data_path = Path().cwd().parents[1] / 'data' / 'interim' / '201708_ODIM'
max_vol2bird_instances = 6 # Set to 1 if you want to disable parallellisation

if max_vol2bird_instances > cpu_count():
    print('Cannot run more vol2bird instances than number of CPU cores available. Lower the number of instances.')
    exit(1)

# Get Docker client
client = docker.from_env()
containers = {}

for docker_instance in range(max_vol2bird_instances):
    try:
        containers[docker_instance] = client.containers.get('vol2bird-{}'.format(docker_instance))
        containers[docker_instance].exec_run('bash -c "cd /data2"', stdout=True, stderr=True).output.decode('utf-8')
        print('The vol2bird-{} Docker container seems to be running already.'.format(docker_instance))
    except docker.errors.NotFound:
        volumes = {data_path: {'bind': '/data', 'mode': 'rw'}}
        containers[docker_instance] = client.containers.run(image='adokter/vol2bird',
                                                           name='vol2bird-{}'.format(docker_instance),
                                                           volumes=volumes, command='sleep infinity', detach=True)
        containers[docker_instance].exec_run('bash -c "cd /data2"', stdout=True, stderr=True).output.decode('utf-8')
        print('Started the vol2bird-{} container'.format(docker_instance))
    except requests.exceptions.ConnectionError as e:
        print('Docker error: {}\n Are you sure the Docker client is running (e.g. Docker for Mac)?'.format(e))

connection = sqlite3.connect('vertical-profiles.db')
cursor = connection.cursor()

sql_table = ('CREATE TABLE IF NOT EXISTS vertical_profiles ('
             'date integer NOT NULL,'
             'time text NOT NULL,'
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
connection.commit()


def run_vol2bird(file):
    vp_path = str(file)
    vp_path = vp_path.replace('_ODIM.h5', '_ODIM.h5.vp.h5')

    if Path(vp_path).is_file():
        print('{} already exists.'.format(vp_path))
        # Apparently vp file already exists, so we move to the next file
        return

    sql_vp = ('INSERT INTO vertical_profiles ('
              'date, time, height, u, v, w, ff, dd, sd_vvp, gap, dbz, eta, dens, DBZH, n, n_dbz, n_all, n_dbz_all)'
              'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);')

    current_worker = int(current_process().name.split('-')[1]) - 1

    connection = sqlite3.connect('vertical-profiles.db', timeout=30)
    cursor = connection.cursor()

    try:

        cmd = 'bash -c "cd /data && vol2bird {} {}.vp.h5"'.format(file.name, file.name)
        out = containers[current_worker].exec_run(cmd, stdout=True, stderr=True).output.decode('utf-8')

        for line in out.splitlines():
            if line.startswith('#') or line.startswith('Warning: '):
                continue

            data = line.split()
            cursor.execute(sql_vp, data)
            connection.commit()

        print(out)
        print('Finished file: {}'.format(file.name))

    except docker.errors.APIError as e:
        print('Docker error: {}'.format(e))

    except sqlite3.ProgrammingError as e:
        print('SQLite error: {}\n While processing file: {}'.format(e, file.name))

    cursor.close()
    connection.close()


with Pool(processes=max_vol2bird_instances) as pool:
    files = [file for file in data_path.glob('*_ODIM.h5')]
    pool.map(run_vol2bird, files)

for container_id in containers:
    try:
        containers[container_id].kill()
        containers[container_id].remove()
        print('Stopped and removed vol2bird-{} Docker container.'.format(container_id))
    except docker.errors.APIError as e:
        print('Docker error: {}\nDid the Docker client stop running while processing files with vol2bird?'.format(e))

print('Elapsed time: {}'.format(time.time() - start_time))



