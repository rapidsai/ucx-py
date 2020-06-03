Using with Dask
===============

``UCX/UCX-Py`` can be used with `Dask <https://dask.org/>`_ as a drop-in replacement for the communication protocol between workers.  Below we show how to use UCX-Py with both helper utilities such as `dask-cuda <https://github.com/rapidsai/dask-cuda>`_
and manually starting a dask cluster with UCX enabled.  Additionally, we demonstrate using UCX with a `cuDF Example`_ and `CuPy Example`_.

Starting with Dask-cuda
-----------------------

`dask-cuda <https://github.com/rapidsai/dask-cuda>`_ is a Python tool designed to help with cluster deployment and management of Dask workers on CUDA-enabled systems.  Dask-cuda can be used to in two ways: inline and CLI.

Inline
~~~~~~

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    from dask_cuda.initialize import initialize

    # ON/OFF settings for various devices
    enable_tcp_over_ucx = True
    enable_infiniband = False
    enable_nvlink = False


    # initialize client with the same settings as workers
    initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    cluster = LocalCUDACluster(
        interface="enp1s0f0",  # Ethernet interface
        protocol="ucx",
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )
    client = Client(cluster)


CLI
~~~~

Dask-cuda can also be used when manually starting a cluster:

.. code-block:: bash

    # server
    # Note: --interface is an Ethernet interface
    UCX_CUDA_IPC_CACHE=n UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,sockcm,cuda_copy,cuda_ipc \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm python -m distributed.cli.dask_scheduler --interface enp1s0f0 --protocol ucx


    # worker
    UCX_CUDA_IPC_CACHE=n UCX_TLS=tcp,sockcm,cuda_copy,cuda_ipc \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm dask-cuda-worker ucx://{SCHEDULER_ADDR}:8786

    # client
    UCX_CUDA_IPC_CACHE=n UCX_TLS=tcp,cuda_copy,cuda_ipc,sockcm \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm python <python file>


The benefit of using ``dask-cuda-worker`` is that it will invoke N workers where N is the number of GPUs and automatically pair workers with GPUs.


Manual Cluster Creation
-----------------------

Lastly, we can also manually start each worker individually (this is typically only used when debugging):

.. code-block:: bash

    # server
    UCX_CUDA_IPC_CACHE=n UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,sockcm,cuda_copy,cuda_ipc \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm python -m distributed.cli.dask_scheduler --interface enp1s0f0 --protocol ucx

    # worker
    CUDA_VISIBLE_DEVICES=0 UCX_CUDA_IPC_CACHE=n UCX_TLS=tcp,sockcm,cuda_copy,cuda_ipc \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm dask-worker ucx://{SCHEDULER_ADDR}:8786

    # client
    UCX_CUDA_IPC_CACHE=n UCX_TLS=tcp,cuda_copy,cuda_ipc,sockcm \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm python <python file>

Note: ``CUDA_VISIBLE_DEVICES`` controls which GPU(s) the worker has access to and ``--interface`` is an Ethernet interface

cuDF Example
------------

.. literalinclude:: ../../examples/cudf-example.py
   :language: python

CuPy Example
------------

.. literalinclude:: ../../examples/cupy-example.py
   :language: python
