## Overview

The goal of the python bindings is to use UCX API for connection
management and data transfer while attempting to be
pythonesque. Specifically, using UCX's UCP listen-connect and
tag_send/recv functionality borrowed heavily from
[ucp_client_server.c](https://github.com/openucx/ucx/blob/master/test/examples/ucp_client_server.c)
and
[ucp_hello_world.c](https://github.com/openucx/ucx/blob/master/test/examples/ucp_hello_world.c).

To ease python-side usage, the bindings also attempt to provide
minimal functionality similar to those familiar with [Future
objects](https://docs.python.org/3/library/concurrent.futures.html#future-objects). Minimally,
the objects returned from send/recv operations can be called with
`done()` or `result()` to query status or get results of transfer
requests. Furthermore, a subset of transfer functions is compatible
with [asyncio](https://docs.python.org/3/library/asyncio.html) style
of programming. For instance, a coroutine can be used at the
server-side to handle incoming connections concurrently with other
coroutines similar to an example
[here](https://asyncio.readthedocs.io/en/latest/tcp_echo.html).

Lastly, some buffer management utilities are provided for transfer of
contiguous python objects, and for simple experiments.

## Functions
1. UCP API Usage
   + Cython definitions + Python bindings
     - [ucp_py.pyx](./ucp_py.pyx)
   + Cython helper layer
     - [ucp_py_ucp_fxns_wrapper.pyx](./ucp_py_ucp_fxns_wrapper.pyx)
   + C definitions
     - [ucp_py_ucp_fxns.c](./ucp_py_ucp_fxns.c)
     - [ucp_py_ucp_fxns.h](./ucp_py_ucp_fxns.h)
     - [common.h](./common.h)
2. Buffer Management
   + Cython helper layer + Python bindings
     - [ucp_py_buffer_helper.pyx](./ucp_py_buffer_helper.pyx)
   + C definitions
     - [buffer_ops.c](./buffer_ops.c)
     - [buffer_ops.h](./buffer_ops.h)
     - [common.h](./common.h)
3. Build
   + [setup.py](./setup.py)

### UCP API Usage

The basic connection model envisioned for UCP python bindings usage is for a
process to call:
 + listen API if it expects connections
   - `.start_server`
 + connect API targeting listening processes
   - `.get_endpoint(server_ip, server_port)`
 + get bidirectional endpoint handles from connections
   - available as part of listen-accept callback @ server
   - returned from `.get_endpoint(server_ip, server_port)` @ client

The envisioned transfer model is to call:
 + send/recv on endpoints
   - `ep.send_msg`, `ep.recv_msg` which return `CommFuture` objects on
     which `.done` or `.result` calls can be made
   - `ep.send_fast`, `ep.recv_fast` which return `ucp_comm_request`
     objects on which *only* `.done` or `.result` calls can be made
   - `ep.send_msg`, `ep.recv_msg` which return `ucp_comm_request`
     object and the received python object respectively
 + optionally make explicit progress on outstanding transfers
   - call `.progress()`

The above calls are exposed through classes/functions in
[ucp_py.pyx](./ucp_py.pyx). The corresponding C backend are written in
[ucp_py_ucp_fxns.c](./ucp_py_ucp_fxns.c) and
[ucp_py_ucp_fxns.h](./ucp_py_ucp_fxns.h)

### Buffer Management

A single (cython?) class **buffer_region** exposes methods to
allocate/free host and cuda buffers. This class is defined in
[ucp_py_buffer_helper.pyx](./ucp_py_buffer_helper.pyx). The allocated
buffers is stored using *data_buf* structure, which in turn stores the
pointer to the allocated buffer (TODO: Simplify -- this seems
roundabout). **buffer_region** can be used with contiguous python
objects without needing to allocate memory using `alloc_host` or
`alloc_cuda` methods. To use pointers pointing to the start of the
python object, the `populate_ptr(python-object)` method can be
used. If a **buffer_region** object is associated with a receive
operation, `return_obj()` method can be used to obtain the contiguous
received object.

Internally, `alloc_*` methods translate to `malloc` or `cudaMalloc`
and `free_*` to their counterparts. For experimentation purpose and
for data-validation checks, there are `set_*_buffer` methods that
initialize allocated buffers with a specific
character. `check_*_buffer` methods return the number of mismatches in
the buffers with the character provided. `set_*_buffer` methods have
not be tested when python objects are assoicated with
**buffer_region's** *data_buf* structure.
