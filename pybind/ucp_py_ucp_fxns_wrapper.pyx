# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

cdef extern from "ucp_py_ucp_fxns.h":
    void ucp_py_worker_progress()
    int ucp_py_init()
    int ucp_py_listen(server_accept_cb_func, void *, int)
    int ucp_py_finalize()
    ucp_ep_h* ucp_py_get_ep(char *, int)
    int ucp_py_put_ep(ucp_ep_h *)
    ucx_context* ucp_py_ep_send_nb(ucp_ep_h*, data_buf*, int)
    ucx_context* ucp_py_recv_nb(data_buf*, int)
    int ucp_py_ep_post_probe()
    int ucp_py_probe_query()
    int ucp_py_probe_wait()
    int ucp_py_query_request(ucx_context*)
