## Functions
1. Buffer Management
   + Cython Layer
     - [ucp_py_buffer_helper.pyx](./ucp_py_buffer_helper.pyx)
   + C definitions
     - [buffer_ops.c](./buffer_ops.c)
     - [buffer_ops.h](./buffer_ops.h)
     - [common.h](./common.h)
2. UCP API Usage
   + Python module API
     - [ucp_py_ucp_fxns_wrapper.pyx](./ucp_py_ucp_fxns_wrapper.pyx)
   + Cython/Python definitions
     - [ucp_py.pyx](./ucp_py.pyx)
   + C definitions
     - [ucp_py_ucp_fxns.c](./ucp_py_ucp_fxns.c)
     - [ucp_py_ucp_fxns.h](./ucp_py_ucp_fxns.h)
     - [common.h](./common.h)
3. Build
   + [setup.py]()
