Configuration
=============

UCX can be configured with a wide variety of options and optimizations including: transport, caching, etc.

For novice users we recommend the following settings:

::

    UCX_MEMTYPE_CACHE=n UCX_TLS=all

``UCX_TLS=all`` instructs UCX to try all available transport methods.  However, users who want to define what transport methods and/or other optional settings may do so.  Below we define the more common environment variables and provide some example combinations and usage.

Env Vars
--------

DEBUG
~~~~~

Debug variables for both UCX and UCX-PY can be set

``UCXPY_LOG_LEVEL``
``UCX_LOG_LEVEL``

Values: DEBUG, TRACE

If UCX has been built with debug mode enabled

MEMORY
~~~~~~

``UCX_MEMTYPE_CACHE``

This is a UCX Memory optimization which toggles whether UCX library intercepts cu*alloc* calls.  UCX-PY defaults this value to  ``n``.  There `known issues <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>`_ when using this feature.

Values: n

``UCX_RNDV_SCHEME``

Communication scheme in RNDV protocol

Values:

- ``put_zcopy``
- ``get_zcopy``
- ``auto`` (default)


``UCX_TLS``

Transport Methods (Simplified):

- ``rc`` -> InfiniBand (ibv_post_send, ibv_post_recv, ibv_poll_cq)
- ``cuda_copy`` -> cuMemHostRegister, cuMemcpyAsync
- ``cuda_ipc`` -> NVLINK (cuIpcCloseMemHandle, cuIpcOpenMemHandle, cuMemcpyAsync)
- ``sockcm`` -> connection management over sockets
- ``tcp`` -> TCP over sockets (often used with sockcm)


InfiniBand Device
~~~~~~~~~~~~~~~~~~

Select InfiniBand Device

``UCX_NET_DEVICES``

Typically these will be the InfiniBand device corresponding to a particular set of GPUs.  Values:

- ``mlx5_0:1``

To find more information on the topology of InfiniBand-GPU pairing run the following::

   nvidia-smi topo -m

Example Configs
---------------

IB -- Yes NVLINK
~~~~~~~~~~~~~~~~

::

    UCX_RNDV_SCHEME=put_zcopy UCX_MEMTYPE_CACHE=n UCX_TLS=rc,cuda_copy,cuda_ipc

TLS/Socket -- No NVLINK
~~~~~~~~~~~~~~~~~~~~~~~

::

    UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,cuda_copy,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm <SCRIPT>

TLS/Socket -- Yes NVLINK
~~~~~~~~~~~~~~~~~~~~~~~~

::

    UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,cuda_copy,cuda_ipc,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm <SCRIPT>
