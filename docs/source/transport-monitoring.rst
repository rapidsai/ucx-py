Monitoring Transports
=====================

Below is a list of commonly used tools and commands to monitor Infiniband and CUDA IPC messages:


Infiniband
----------

Montior Infiniband packet counters -- this number should dramatically increase when data is sent via IB:

::

    watch -n 0.1 'cat /sys/class/infiniband/mlx5_*/ports/1/counters/port_xmit_data'


CUDA IPC/NVLink
---------------

Montior traffic over all GPUs on counter 0

::

    # set counters
    nvidia-smi nvlink -sc 0bz
    watch -d 'nvidia-smi nvlink -g 0'

Stats Monitoring of GPUs
::

    dcgmi dmon -e 449

nvdashboard
