Send/Recv Internals
===================

Generally UCX creates connections between endpoints with the following steps:

1. Create a ``Listener`` with defined IP address and port
  a. ``Listener`` defines a callback function to process communications from endpoints
2. Connect an ``Endpoint`` to the ``Listener``
3. ``Endpoint`` sends and receives data with the ``Listener``


Below we go into more detail as we create an echo server in UCX and compare with `Python Sockets <https://docs.python.org/3/library/socket.html#example>`_

Server
------
First, we create the server -- in UCX-Py, we create a server with ``create_listener`` and build a blocking call to keep the listener alive.  The listener invokes a callback function when an incoming connection is accepted.  This callback should take in an ``Endpoint`` as an argument for send/recv.

For Python sockets, the server is similarly constructed. ``bind`` opens a connection on a given port and ``accept`` is Python Sockets' blocking call for incoming connections.  In both UCX-Py and Sockets, once a connection has been made, both receive data and echo the same data back to the client

+------------------------------------------------------+----------------------------------------------------------+
| UCX                                                  | Python Sockets                                           |
+------------------------------------------------------+----------------------------------------------------------+
| .. code-block:: python                               | .. code-block:: python                                   |
|                                                      |                                                          |
|    async def echo_server(ep):                        |     s = socket.socket(...)                               |
|        while True:                                   |                                                          |
|            obj = await ep.recv_obj()                 |     s.bind((HOST, PORT))                                 |
|            await ep.send_obj(obj)                    |     s.listen(1)                                          |
|                                                      |     conn, addr = s.accept()                              |
|    lf = ucp.create_listener(echo_server, port)       |                                                          |
|                                                      |     while True:                                          |
|    while not lf.closed():                            |         data = conn.recv(1024)                           |
|        await asyncio.sleep(0.1)                      |         if not data: break                               |
|                                                      |         conn.sendall(data)                               |
+------------------------------------------------------+----------------------------------------------------------+


Client
------

For Sockets, on the client-side we connect to the established host/port combination and send data to the socket.  Whereas in UCX-Py, the client-side is a bit more interesting.  ``create_endpoint``, also uses a host/port combination to establish a connection, and after an ``Endpoint`` is created, ``hello, world`` is passed back and forth between the client an server.

+------------------------------------------------------+----------------------------------------------------------+
| UCX                                                  | Python Sockets                                           |
+------------------------------------------------------+----------------------------------------------------------+
| .. code-block:: python                               | .. code-block:: python                                   |
|                                                      |                                                          |
|    client = await ucp.create_endpoint(addr, port)    |    s = socket.socket(...)                                |
|                                                      |    s.connect((HOST, PORT))                               |
|    msg = bytearray(b"hello, world")                  |    s.sendall(b'hello, world')                            |
|    await client.send_obj(msg)                        |    echo_msg = s.recv(1024)                               |
|    echo_msg = await client.recv_obj()                |                                                          |
|                                                      |                                                          |
+------------------------------------------------------+----------------------------------------------------------+

So what happens with ``create_endpoint`` ?  UCX, unlike with Sockets, employs a tag-matching strategy where endpoints are created with a unique id and send/receive operations also use unique tags for those. For more details on tag-matching please see the `following page <https://community.mellanox.com/s/article/understanding-tag-matching-for-developers>`_. ``create_endpoint``, will create an ``Endpoint`` with three steps:

#. Generate unique IDs to use as tags
#. Exchange endpoint info such as tags
#. Use the info to create an endpoint

Again, an ``Endpoint`` sends and receives with `unique tags <http://openucx.github.io/ucx/api/v1.8/html/group___u_c_t___t_a_g.html>`_.

.. code-block:: python

    ep = Endpoint(
            endpoint=ucx_ep,
            ctx=self,
            msg_tag_send=peer_info["msg_tag"],
            msg_tag_recv=msg_tag,
            ctrl_tag_send=peer_info["ctrl_tag"],
            ctrl_tag_recv=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order,
        )

Most users will should not care about these details but developers and interested network enthusiasts may.  Looking at the DEBUG (``UCXPY_LOG_LEVEL=DEBUG``) output of the client can help illuminate what UCX-Py/UCX is doing::


    # client = await ucp.create_endpoint(addr, port)
    [1594305203.642096] [dgx12:67497] UCXPY  DEBUG create_endpoint() client: 0x7fef8bdc8048, msg-tag-send: 0x40b7487ebf0edc3, msg-tag-recv: 0xabeef1db009f97af, ctrl-tag-send: 0xee742ff94899db5c, ctrl-tag-recv: 0xa2c3145e1eec9b86

    # await client.send_obj(msg)
    [1594305218.811759] [dgx12:67497] UCXPY  DEBUG [Send #000] ep: 0x7fef8bdc8048, tag: 0x40b7487ebf0edc3, nbytes: 8, type: <class 'bytes'>
    [1594305218.812014] [dgx12:67497] UCXPY  DEBUG [Send #001] ep: 0x7fef8bdc8048, tag: 0x40b7487ebf0edc3, nbytes: 12, type: <class 'bytearray'>

    # echo_msg = await client.recv_obj()
    [1594305226.988246] [dgx12:67497] UCXPY  DEBUG [Recv #000] ep: 0x7fef8bdc8048, tag: 0xabeef1db009f97af, nbytes: 8, type: <class 'bytearray'>
    [1594305226.989332] [dgx12:67497] UCXPY  DEBUG [Recv #001] ep: 0x7fef8bdc8048, tag: 0xabeef1db009f97af, nbytes: 12, type: <class 'bytearray'>


We can see from the above that when the ``Endpoint`` is created, 4 tags are generated:  ``msg-tag-send``, ``msg-tag-recv``, ``ctrl-tag-send``, and ``ctrl-tag-recv``.  This data is transmitted to the server via a `stream <http://openucx.github.io/ucx/api/v1.8/html/group___u_c_p___c_o_m_m.html#ga9022ff0ebb56cac81f6ba81bb28f71b3>`_ send/receive in an `exchange peer info <https://github.com/rapidsai/ucx-py/blob/6e1c1d201a382c689ca098c848cbfdc8237e1eba/ucp/core.py#L38-L89>`_ convenience function.

Next, The client sends data on the ``msg-tag-send`` tag.  Two messages are sent, the size of the data ``8 bytes`` and data itself.  The server receives the data and immediately echos the data back.  Lastly, the client the receives two messages the size of the data an the data itself
