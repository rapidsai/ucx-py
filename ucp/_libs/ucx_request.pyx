# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from cpython.ref cimport Py_DECREF, Py_INCREF, PyObject
from libc.stdint cimport uintptr_t

from .ucx_api_dep cimport *

from ..exceptions import UCXError, UCXMsgTruncated


# Counter used as UCXRequest UIDs
cdef unsigned int _ucx_py_request_counter = 0


cdef class UCXRequest:
    """Python wrapper of UCX request handle.

    Don't create this class directly, the send/recv functions and their
    callback functions will return UCXRequest objects.

    Notice, this class doesn't own the handle and multiple instances of
    UCXRequest can point to the same underlying UCX handle.
    Furthermore, UCX can modify/free the UCX handle without notice
    thus we use `_uid` to make sure the handle hasn't been modified.
    """
    cdef:
        ucx_py_request *_handle
        unsigned int _uid

    def __init__(self, uintptr_t req_as_int):
        global _ucx_py_request_counter
        cdef ucx_py_request *req = <ucx_py_request*>req_as_int
        assert req != NULL
        self._handle = req

        cdef dict info = {"status": "pending"}
        if self._handle.info == NULL:  # First time we are wrapping this UCX request
            Py_INCREF(info)
            self._handle.info = <PyObject*> info
            _ucx_py_request_counter += 1
            self._uid = _ucx_py_request_counter
            assert self._handle.uid == 0
            self._handle.uid = _ucx_py_request_counter
        else:
            self._uid = self._handle.uid

    cpdef bint closed(self):
        return self._handle == NULL or self._uid != self._handle.uid

    cpdef void close(self) except *:
        """This routine releases the non-blocking request back to UCX,
        regardless of its current state. Communications operations associated with
        this request will make progress internally, however no further notifications or
        callbacks will be invoked for this request. """

        if not self.closed():
            Py_DECREF(<object>self._handle.info)
            self._handle.info = NULL
            self._handle.uid = 0
            ucp_request_free(self._handle)
            self._handle = NULL

    @property
    def info(self):
        assert not self.closed()
        return <dict> self._handle.info

    @property
    def handle(self):
        assert not self.closed()
        return int(<uintptr_t>self._handle)

    def __hash__(self):
        if self.closed():
            return id(self)
        else:
            return self._uid

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        if self.closed():
            return "<UCXRequest closed>"
        else:
            return (
                f"<UCXRequest handle={hex(self.handle)} "
                f"uid={self._uid} info={self.info}>"
            )


cdef UCXRequest _handle_status(
    ucs_status_ptr_t status,
    int64_t expected_receive,
    cb_func,
    cb_args,
    cb_kwargs,
    unicode name,
    set inflight_msgs
):
    if UCS_PTR_STATUS(status) == UCS_OK:
        return
    cdef str ucx_status_msg, msg
    if UCS_PTR_IS_ERR(status):
        ucx_status_msg = (
            ucs_status_string(UCS_PTR_STATUS(status)).decode("utf-8")
        )
        msg = "<%s>: %s" % (name, ucx_status_msg)
        raise UCXError(msg)
    cdef UCXRequest req = UCXRequest(<uintptr_t><void*> status)
    assert not req.closed()
    cdef dict req_info = <dict>req._handle.info
    if req_info["status"] == "finished":
        try:
            # The callback function has already handled the request
            received = req_info.get("received", None)
            if received is not None and received != expected_receive:
                msg = "<%s>: length mismatch: %d (got) != %d (expected)" % (
                    name, received, expected_receive
                )
                raise UCXMsgTruncated(msg)
            else:
                cb_func(req, None, *cb_args, **cb_kwargs)
                return
        finally:
            req.close()
    else:
        req_info["cb_func"] = cb_func
        req_info["cb_args"] = cb_args
        req_info["cb_kwargs"] = cb_kwargs
        req_info["expected_receive"] = expected_receive
        req_info["name"] = name
        inflight_msgs.add(req)
        req_info["inflight_msgs"] = inflight_msgs
        return req
