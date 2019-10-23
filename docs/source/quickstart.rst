Quickstart
==========


Install
-------

Conda
~~~~~

::

    conda create -n ucx -c conda-forge -c jakirkham/label/ucx \
    cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7

Source
~~~~~~

The following instructions assume you'll be using ucx-py on a CUDA enabled system.

Note: UCX depends on the following system libraries being present: ``libibverbs``, ``librdmacm``, and ``libnuma`` (numactl on Enterprise Linux).  Please install these with your Linux system's package manager.



1) Install UCX

::

    git clone https://github.com/openucx/ucx
    cd ucx
    ./autogen.sh
    mkdir build
    cd build
    ../configure --prefix=$CONDA_PREFIX --enable-debug --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I//$CUDA_HOME/include"
    make -j install

2) Install UCX-PY

::

    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    46python setup.py build_ext --inplace
    python -m pip install -e .


Send/Recv CuPy Arrays
---------------------

Process 1
~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucp
    import numpy as np

    n_bytes = 100
    host = ucp.get_address(ifname='enp1s0f0')
    port = 13337

    async def send(ep):
        # recv buffer
        arr = cp.empty(n_bytes, dtype=cp.uint8)
        await ep.recv(arr)
        print("Received CuPy array")

        # increment array and send back
        arr += 1
        print("Sending incremented CuPy array")
        await ep.send(arr)

        await ep.signal_shutdown()
        ep.close()
        lf.close()

    lf = ucp.create_listener(send, port)
    while not lf.closed():
        await asyncio.sleep(0.1)



Process 2
~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucp
    import cupy as cp
    import numpy as np

    port = 13337
    host = ucp.get_address(ifname='enp1s0f0')
    ep = await ucp.create_endpoint(host, port)
    msg = cp.zeros(100, dtype='u1') # create some data to send
    msg_size = np.array([msg.nbytes], dtype=np.uint64)

    # send message
    print("Send Original CuPy array")
    await ep.send(msg, msg_size)  # send the real message

    # recv response
    print("Receive Incremented CuPy arrays")
    resp = cp.empty_like(msg)
    await ep.recv(resp, msg_size)  # receive the echo
    cp.testing.assert_array_equal(msg + 1, resp)