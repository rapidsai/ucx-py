Configuration
=============

Env Vars
----------

DEBUG
~~~~~


``UCX_PY_LOG_LEVEL``
``UCX_LOG_LEVEL``

Values: DEBUG, TRACE

MEMORY
~~~~~~

``UCX_MEMTYPE_CACHE``

UCX Memory optimization `known issues <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>`_.  UCX-PY regularly sets this to `n` -- toggles whether UCX library intercepts cu*alloc* calls.

Values: n

```UCX_RNDV_SCHEME``

Values: put_zcopy


``UCX_TLS``

Values:
- rc = ibv_post_send, ibv_post_recv, ibv_poll_cq
- cuda_copy = cuMemHostRegister, cuMemcpyAsync
- cuda_ipc =  cuIpcCloseMemHandle ,  cuIpcOpenMemHandle, cuMemcpyAsync
- sockcm = connection management over sockets
- tcp = communication over TCP


Example Usages
--------------

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
