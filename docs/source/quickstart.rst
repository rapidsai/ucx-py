Quickstart
==========


Setup
-----

With GPU support:

::

    conda create -n ucx -c conda-forge -c rapidsai \
      cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7

Without GPU support:

::

    conda create -n ucx -c conda-forge -c rapidsai \
      ucx-proc=*=cpu ucx ucx-py python=3.7

For a more detailed guide on installation options please refer to the :doc:`install` page.

Send/Recv NumPy Arrays
---------------------

Process 1 - Server
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import time
    import ucp
    import numpy as np

    n_bytes = 2**30
    host = ucp.get_address(ifname='eth0')  # ethernet device name
    port = 13337

    async def send(ep):
        # recv buffer
        arr = np.empty(n_bytes, dtype='u1')
        await ep.recv(arr)
        assert np.count_nonzero(arr) == np.array(0, dtype=np.int64)
        print("Received CuPy array")

        # increment array and send back
        arr += 1
        print("Sending incremented NumPy array")
        await ep.send(arr)

        await ep.close()
        lf.close()

    async def main():
        global lf
        lf = ucp.create_listener(send, port)

        while not lf.closed():
            await asyncio.sleep(0.1)

    if __name__ == '__main__':
        asyncio.run(main())

Process 2 - Client
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucp
    import numpy as np

    port = 13337
    n_bytes = 2**30

    async def main():
        host = ucp.get_address(ifname='eth0')  # ethernet device name
        ep = await ucp.create_endpoint(host, port)
        msg = np.zeros(n_bytes, dtype='u1') # create some data to send
        msg_size = np.array([msg.nbytes], dtype=np.uint64)

        # send message
        print("Send Original NumPy array")
        await ep.send(msg, msg_size)  # send the real message

        # recv response
        print("Receive Incremented NumPy arrays")
        resp = np.empty_like(msg)
        await ep.recv(resp, msg_size)  # receive the echo
        await ep.close()
        np.testing.assert_array_equal(msg + 1, resp)

    if __name__ == '__main__':
        asyncio.run(main())



Send/Recv CuPy Arrays
---------------------

.. note::
    If you are passing CuPy arrays between GPUs and want to use `NVLINK <https://www.nvidia.com/en-us/data-center/nvlink/>`_ ensure you have correctly set ``UCX_TLS`` with ``cuda_ipc``. See the :doc:`configuration` for more details

Process 1 - Server
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import time
    import ucp
    import cupy as cp

    n_bytes = 2**30
    host = ucp.get_address(ifname='eth0')  # ethernet device name
    port = 13337

    async def send(ep):
        # recv buffer
        arr = cp.empty(n_bytes, dtype='u1')
        await ep.recv(arr)
        assert cp.count_nonzero(arr) == cp.array(0, dtype=cp.int64)
        print("Received CuPy array")

        # increment array and send back
        arr += 1
        print("Sending incremented CuPy array")
        await ep.send(arr)

        await ep.close()
        lf.close()

    async def main():
        global lf
        lf = ucp.create_listener(send, port)

        while not lf.closed():
            await asyncio.sleep(0.1)

    if __name__ == '__main__':
        asyncio.run(main())

Process 2 - Client
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucp
    import cupy as cp
    import numpy as np

    port = 13337
    n_bytes = 2**30

    async def main():
        host = ucp.get_address(ifname='eth0')  # ethernet device name
        ep = await ucp.create_endpoint(host, port)
        msg = cp.zeros(n_bytes, dtype='u1') # create some data to send
        msg_size = np.array([msg.nbytes], dtype=np.uint64)

        # send message
        print("Send Original CuPy array")
        await ep.send(msg, msg_size)  # send the real message

        # recv response
        print("Receive Incremented CuPy arrays")
        resp = cp.empty_like(msg)
        await ep.recv(resp, msg_size)  # receive the echo
        await ep.close()
        cp.testing.assert_array_equal(msg + 1, resp)

    if __name__ == '__main__':
        asyncio.run(main())
