# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import functools
import logging

from libc.stdint cimport uintptr_t
from libc.stdio cimport FILE
from libc.string cimport memset

from .ucx_api_dep cimport *

logger = logging.getLogger("ucx")


def _ucx_context_handle_finalizer(uintptr_t handle):
    ucp_cleanup(<ucp_context_h> handle)


cdef class UCXContext(UCXObject):
    """Python representation of `ucp_context_h`

    Parameters
    ----------
    config_dict: Mapping[str, str]
        UCX options such as "MEMTYPE_CACHE=n" and "SEG_SIZE=3M"
    feature_flags: Iterable[Feature]
        Tuple of UCX feature flags
    """
    cdef:
        ucp_context_h _handle
        dict _config
        tuple _feature_flags
        readonly bint cuda_support

    def __init__(
        self,
        config_dict={},
        feature_flags=(
            Feature.TAG,
            Feature.WAKEUP,
            Feature.STREAM,
            Feature.AM,
            Feature.RMA
        )
    ):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        self._feature_flags = tuple(feature_flags)

        memset(&ucp_params, 0, sizeof(ucp_params))
        ucp_params.field_mask = (
            UCP_PARAM_FIELD_FEATURES |
            UCP_PARAM_FIELD_REQUEST_SIZE |
            UCP_PARAM_FIELD_REQUEST_INIT
        )
        ucp_params.features = functools.reduce(
            lambda x, y: x | y.value, feature_flags, 0
        )
        ucp_params.request_size = sizeof(ucx_py_request)
        ucp_params.request_init = (
            <ucp_request_init_callback_t>ucx_py_request_reset
        )

        cdef ucp_config_t *config = _read_ucx_config(config_dict)
        try:
            status = ucp_init(&ucp_params, config, &self._handle)
            assert_ucs_status(status)
            self._config = ucx_config_to_dict(config)
        finally:
            ucp_config_release(config)

        # UCX supports CUDA if "cuda" is part of the TLS or TLS is "all"
        cdef str tls = self._config["TLS"]
        cuda_transports = {"cuda", "cuda_copy"}
        if tls.startswith("^"):
            # UCX_TLS=^x,y,z means "all \ {x, y, z}"
            disabled = set(tls[1:].split(","))
            self.cuda_support = not (disabled & cuda_transports)
        else:
            enabled = set(tls.split(","))
            self.cuda_support = bool(
                enabled & ({"all", "cuda_ipc"} | cuda_transports)
            )

        self.add_handle_finalizer(
            _ucx_context_handle_finalizer,
            int(<uintptr_t>self._handle)
        )

        logger.info("UCP initiated using config: ")
        cdef str k, v
        for k, v in self._config.items():
            logger.info(f"  {k}: {v}")

    cpdef dict get_config(self):
        return self._config

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t>self._handle)

    def info(self):
        assert self.initialized

        cdef FILE *text_fd = create_text_fd()
        ucp_context_print_info(self._handle, text_fd)
        return decode_text_fd(text_fd)

    def map(self, mem):
        return UCXMemoryHandle.map(self, mem)

    def alloc(self, size):
        return UCXMemoryHandle.alloc(self, size)
