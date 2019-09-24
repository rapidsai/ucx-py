import os
import pytest
import subprocess

normal_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0",
}


def set_env():
    os.environ.update(normal_env)


def get_device(device_type='tcp'):
    p = subprocess.Popen("/sbin/ifconfig -a | sed 's/[ \t].*//;/^$/d'", stdout=subprocess.PIPE, shell=True)
    res, err = p.communicate()
    res = res.decode('utf-8')
    devices = res.split()

    for d in devices:
        if device_type == 'tcp':
            if d.startswith('ib') or d.startswith('docker') or d == 'lo':
                continue
            else:
                return d
        if device_type == 'ib':
            if d.startswith('ib'):
                return d

@pytest.fixture
def device_name():
    if 'tcp' in os.environ.get('UCX_TLS', ''):
        return get_device('tcp')
    else:
        return get_device('ib')