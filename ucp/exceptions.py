# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


class UCXBaseException(Exception):
    pass


class UCXError(UCXBaseException):
    pass


class UCXWarning(UserWarning):
    pass


class UCXCloseError(UCXBaseException):
    pass


class UCXCanceled(UCXBaseException):
    pass
