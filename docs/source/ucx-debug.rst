UCX Debugging
=============



System Configuration
--------------------

``ibdev2netdev`` -- check to ensure at least one IB controller is configured for IPoIB

::
    user@mlnx:~$ ibdev2netdev
    mlx5_0 port 1 ==> ib0 (Up)
    mlx5_1 port 1 ==> ib1 (Up)
    mlx5_2 port 1 ==> ib2 (Up)
    mlx5_3 port 1 ==> ib3 (Up)


ucx_perftest
------------

Perf test for IB should be in the 10+ GB/s range
::

    CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,cuda_copy ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 & \
    CUDA_VISIBLE_DEVICES= 1UCX_NET_DEVICES=mlx5_1:1 UCX_TLS=rc,cuda_copy ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999

    +--------------+-----------------------------+---------------------+-----------------------+
    |              |       latency (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    | # iterations | typical | average | overall |  average |  overall |   average |   overall |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    +------------------------------------------------------------------------------------------+
    | API:          protocol layer                                                             |
    | Test:         tag match bandwidth                                                        |
    | Data layout:  (automatic)                                                                |
    | Send memory:  cuda                                                                       |
    | Recv memory:  cuda                                                                       |
    | Message size: 100000000                                                                  |
    +------------------------------------------------------------------------------------------+
                10     0.000  9104.800  9104.800   10474.41   10474.41         110         110


Perf test for NVLink should be in the 20+ GB/s range
::

    CUDA_VISIBLE_DEVICES=0 UCX_TLS=cuda_ipc,cuda_copy,tcp,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm  ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 -c 0 & \
    CUDA_VISIBLE_DEVICES=1 UCX_TLS=cuda_ipc,cuda_copy,tcp,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999 -c 1
    +--------------+-----------------------------+---------------------+-----------------------+
    |              |       latency (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    | # iterations | typical | average | overall |  average |  overall |   average |   overall |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    +------------------------------------------------------------------------------------------+
    | API:          protocol layer                                                             |
    | Test:         tag match bandwidth                                                        |
    | Data layout:  (automatic)                                                                |
    | Send memory:  cuda                                                                       |
    | Recv memory:  cuda                                                                       |
    | Message size: 100000000                                                                  |
    +------------------------------------------------------------------------------------------+
                10     0.000  4163.694  4163.694   22904.52   22904.52         240         240


GOTCHAS!
--------

Things we have run into along the way::

- System-wide settings: ``env|grep ^UCX_``.  For example, we saw a system with ``UCX_MEM_MMAP_HOOK_MODE`` set to ``none``.  Unsetting this env var resolved problems:
https://github.com/rapidsai/ucx-py/issues/616