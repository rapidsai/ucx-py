# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import contextlib
import logging

logger = logging.getLogger("ucx")


@contextlib.contextmanager
def log_errors(reraise_exception=False):
    try:
        yield
    except BaseException as e:
        logger.exception(e)
        if reraise_exception:
            raise


class UCXBaseException(Exception):
    pass


class UCXError(UCXBaseException):
    pass


class UCXConfigError(UCXError):
    pass


class UCXWarning(UserWarning):
    pass


class UCXCloseError(UCXBaseException):
    pass


class UCXCanceled(UCXBaseException):
    pass


class UCXMsgTruncated(UCXBaseException):
    pass
