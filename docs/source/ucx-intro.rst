Send/Recv Internals
===================

Generally UCX creates connections between endpoints with the following steps:

1. Create a ``Listener`` with defined ip address an port
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
|        obj = await ep.recv_obj()                     |     s.bind((HOST, PORT))                                 |
|        await ep.send_obj(obj)                        |     s.listen(1)                                          |
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

Most users will should not care about these details but developer may.  To more clearly let's look at the DEBUG (UCXPY_LOG_LEVEL_DEBUG) output of the client::

    [1594232644.118938] [dgx12:23538] UCXPY  DEBUG [Send #006] ep: 0x7f64cc547048, tag: 0x6be7deaa73c05d68, nbytes: 8, type: <class 'numpy.ndarray'>
    [1594232644.119083] [dgx12:23538] UCXPY  DEBUG [Send #007] ep: 0x7f64cc547048, tag: 0x6be7deaa73c05d68, nbytes: 4, type: <class 'numpy.ndarray'>
    [1594232644.119191] [dgx12:23538] UCXPY  DEBUG [Send #008] ep: 0x7f64cc547048, tag: 0x6be7deaa73c05d68, nbytes: 32, type: <class 'numpy.ndarray'>
    [1594232644.119303] [dgx12:23538] UCXPY  DEBUG [Send #009] ep: 0x7f64cc547048, tag: 0x6be7deaa73c05d68, nbytes: 1, type: <class 'numpy.ndarray'>
    [1594232644.119391] [dgx12:23538] UCXPY  DEBUG [Send #010] ep: 0x7f64cc547048, tag: 0x6be7deaa73c05d68, nbytes: 196, type: <class 'numpy.ndarray'>
    [1594232644.119478] [dgx12:23538] UCXPY  DEBUG [Send #011] ep: 0x7f64cc547048, tag: 0x6be7deaa73c05d68, nbytes: 8388608, type: <class 'numpy.ndarray'>

    [1594232644.121825] [dgx12:23538] UCXPY  DEBUG [Recv #012] ep: 0x7f64cc547048, tag: 0x6529fadf7d9e4c3b, nbytes: 8, type: <class 'numpy.ndarray'>
    [1594232644.124372] [dgx12:23538] UCXPY  DEBUG [Recv #013] ep: 0x7f64cc547048, tag: 0x6529fadf7d9e4c3b, nbytes: 4, type: <class 'numpy.ndarray'>
    [1594232644.124552] [dgx12:23538] UCXPY  DEBUG [Recv #014] ep: 0x7f64cc547048, tag: 0x6529fadf7d9e4c3b, nbytes: 32, type: <class 'numpy.ndarray'>
    [1594232644.124656] [dgx12:23538] UCXPY  DEBUG [Recv #015] ep: 0x7f64cc547048, tag: 0x6529fadf7d9e4c3b, nbytes: 1, type: <class 'numpy.ndarray'>
    [1594232644.124829] [dgx12:23538] UCXPY  DEBUG [Recv #016] ep: 0x7f64cc547048, tag: 0x6529fadf7d9e4c3b, nbytes: 196, type: <class 'numpy.ndarray'>
    [1594232644.124929] [dgx12:23538] UCXPY  DEBUG [Recv #017] ep: 0x7f64cc547048, tag: 0x6529fadf7d9e4c3b, nbytes: 8388608, type: <class 'numpy.ndarray'>

    Something something.....