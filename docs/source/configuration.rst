Configuration
=============

UCX/UCX-Py can be configured with a wide variety of options and optimizations including: transport, caching, etc.  Users can configure
UCX/UCX-Py either with environment variables or programmatically during initialization.  Below we demonstrate setting ``UCX_MEMTYPE_CACHE`` to
``n`` and checking the configuration:

.. code-block:: python

    import ucp
    options = {"MEMTYPE_CACHE": "n"}
    ucp.init(options)
    assert ucp.get_config()['MEMTYPE_CACHE'] is 'n'

.. note::
    When programmatically configuring UCX-Py, the ``UCX`` prefix is not used.

For novice users we recommend the following settings:

::

    UCX_MEMTYPE_CACHE=n UCX_TLS=all

``UCX_TLS=all`` configures UCX to try all available transport methods.  However, users who want to define specific transport methods to use and/or other optional settings may do so.  Below we define the more common options and provide some example combinations and usage.

Env Vars
--------

DEBUG
~~~~~

Debug variables for both UCX and UCX-Py can be set

UCXPY_LOG_LEVEL/UCX_LOG_LEVEL
`````````````````````````````

Values: DEBUG, TRACE

If UCX has been built with debug mode enabled

MEMORY
~~~~~~

UCX_MEMTYPE_CACHE
`````````````````

This is a UCX Memory optimization which toggles whether UCX library intercepts cu*alloc* calls.  UCX-Py defaults this value to  ``n``.  There `known issues <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>`_ when using this feature.

Values: n/y

UCX_CUDA_IPC_CACHE
``````````````````

This is a UCX CUDA Memory optimization which enables/disables a remote endpoint IPC memhandle mapping cache. UCX/UCX-Py defaults this value to ``y``

Values: n/y

UCX_RNDV_THRESH
```````````````

This is a configurable parameter used by UCX to help determine which transport method should be used.  For example, on machines with multiple GPUs, and with NVLink enabled, UCX can deliver messages either through TCP or NVLink.  Sending GPU buffers over TCP is costly as it triggers a device-to-host on the sender side, and then host-to-device transfer on the receiver side --  we want to avoid these kinds of transfers when NVLink is available.  If a buffer is below the threshold, `Rendezvous-Protocol <https://github.com/openucx/ucx/wiki/Rendezvous-Protocol>`_ is triggered and for UCX-Py users, this will typically mean messages will be delivered through TCP.  Depending on the application, messages can be quite small, therefore, we recommend setting a small value if the application uses NVLink or InfiniBand: ``UCX_RNDV_THRESH=8192``

Values: Int (UCX-Py default : 8192)


UCX_RNDV_SCHEME
```````````````

Communication scheme in RNDV protocol

Values:

- ``put_zcopy``
- ``get_zcopy``
- ``auto`` (default)

UCX_TCP_RX_SEG_SIZE
```````````````````

Size of send copy-out buffer when receiving.  This environment variable controls the size of the buffer on the host when receiving data over TCP.

UCX_TCP_TX_SEG_SIZE
```````````````````

Size of send copy-out buffer when transmitting.  This environment variable controls the size of the buffer on the host when sending data over TCP.

UCX-Py uses ``8M`` as the default value for both RX/TX.

.. note::
    Users should take care to properly tune ``UCX_TCP_{RX/TX}_SEG_SIZE`` parameters when mixing TCP with other transports methods as well as when
    using TCP over UCX in isolation.  These variables will impact CUDA transfers when no NVLink or InfiniBand is available between UCX-Py processes.
    These parameters will cause the HostToDevice and DeviceToHost copies of buffers to be broken down in several
    chunks when the size of a buffer exceeds the size defined by these two variables. If an application is expected to transfer very
    large buffers, increasing such values may improve overall performance.

UCX_TLS
```````

Transport Methods (Simplified):

- ``all`` -> use all the available transports
- ``rc`` -> InfiniBand (ibv_post_send, ibv_post_recv, ibv_poll_cq) uses rc_v and rc_x (preferably if available)
- ``cuda_copy`` -> cuMemHostRegister, cuMemcpyAsync
- ``cuda_ipc`` -> CUDA Interprocess Communication (cuIpcCloseMemHandle, cuIpcOpenMemHandle, cuMemcpyAsync)
- ``sockcm`` -> connection management over sockets
- ``sm/shm`` -> all shared memory transports (mm, cma, knem)
- ``mm`` -> shared memory transports - only memory mappers
- ``ugni`` -> ugni_smsg and ugni_rdma (uses ugni_udt for bootstrap)
- ``ib`` -> all infiniband transports (rc/rc_mlx5, ud/ud_mlx5, dc_mlx5)
- ``rc_v`` -> rc verbs (uses ud for bootstrap)
- ``rc_x`` -> rc with accelerated verbs (uses ud_mlx5 for bootstrap)
- ``ud_v`` -> ud verbs
- ``ud_x`` -> ud with accelerated verbs
- ``ud`` -> ud_v and ud_x (preferably if available)
- ``dc/dc_x`` -> dc with accelerated verbs
- ``tcp`` -> sockets over TCP/IP
- ``cuda`` -> CUDA (NVIDIA GPU) memory support
- ``rocm`` -> ROCm (AMD GPU) memory support

SOCKADDR_TLS_PRIORITY
`````````````````````

Priority of sockaddr transports


InfiniBand Device
~~~~~~~~~~~~~~~~~~

Select InfiniBand Device

UCX_NET_DEVICES
```````````````

Typically these will be the InfiniBand device corresponding to a particular set of GPUs.  Values:

- ``mlx5_0:1``

To find more information on the topology of InfiniBand-GPU pairing run the following::

   nvidia-smi topo -m

Example Configs
---------------

InfiniBand -- No NVLink
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    UCX_RNDV_SCHEME=get_zcopy UCX_MEMTYPE_CACHE=n UCX_TLS=rc,tcp,sockcm,cuda_copy UCX_SOCKADDR_TLS_PRIORITY=sockcm <SCRIPT>

InfiniBand -- With NVLink
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    UCX_RNDV_SCHEME=get_zcopy UCX_MEMTYPE_CACHE=n UCX_TLS=rc,tcp,sockcm,cuda_copy,cuda_ipc UCX_SOCKADDR_TLS_PRIORITY=sockcm <SCRIPT>

TLS/Socket -- No NVLink
~~~~~~~~~~~~~~~~~~~~~~~

::

    UCX_RNDV_SCHEME=get_zcopy UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,sockcm,cuda_copy UCX_SOCKADDR_TLS_PRIORITY=sockcm <SCRIPT>

TLS/Socket -- With NVLink
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    UCX_RNDV_SCHEME=get_zcopy UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,sockcm,cuda_copy,cuda_ipc UCX_SOCKADDR_TLS_PRIORITY=sockcm <SCRIPT>
