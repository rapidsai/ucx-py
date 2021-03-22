import enum
from typing import Callable, Iterable, Mapping, Union

def get_current_options() -> None: ...

class UCXObject:
    def close(self) -> None: ...

class Feature(enum.Enum):
    TAG: int
    RMA: int
    AMO32: int
    AMO64: int
    WAKEUP: int
    STREAM: int
    AM: int

class UCXContext(UCXObject):
    def __init__(
        self, config_dict: Mapping = ..., feature_flags: Iterable[Feature] = ...
    ): ...

class UCXAddress:
    @classmethod
    def from_buffer(cls, buffer) -> UCXAddress: ...
    @classmethod
    def from_worker(cls, worker: UCXWorker) -> UCXAddress: ...
    @property
    def address(self) -> int: ...
    @property
    def length(self) -> int: ...

class UCXWorker(UCXObject):
    def __init__(self, context: UCXContext): ...
    def progress(self) -> None: ...
    def ep_create(
        self, ip_address: str, port: int, endpoint_error_handling: bool
    ) -> UCXEndpoint: ...
    def ep_create_from_worker_address(
        self, ip_address: str, port: int, endpoint_error_handling: bool
    ) -> UCXEndpoint: ...
    def ep_create_from_conn_request(
        self, conn_request: int, endpoint_error_handling: bool
    ) -> UCXEndpoint: ...

class UCXListener(UCXObject):
    port: int
    ip: str
    def __init__(
        self,
        worker: UCXWorker,
        port: int,
        cb_func: Callable,
        cb_args: tuple = None,
        cb_kwargs: dict = None,
    ): ...

class UCXEndpoint(UCXObject):
    def info(self) -> str: ...
    @property
    def worker(self) -> UCXWorker: ...

def tag_send_nb(
    ep: UCXEndpoint,
    buffer,
    nbytes: int,
    tag: int,
    cb_func: Callable,
    cb_args: tuple = None,
    cb_kwargs: dict = None,
    name: str = None,
): ...
def tag_recv_nb(
    worker: UCXWorker,
    buffer,
    nbytes: int,
    tag: int,
    cb_func: Callable,
    cb_args: tuple = None,
    cb_kwargs: dict = None,
    name: str = None,
    ep: UCXEndpoint = None,
): ...
def stream_send_nb(
    ep: UCXEndpoint,
    buffer,
    nbytes: int,
    cb_func: Callable,
    cb_args: tuple = None,
    cb_kwargs: dict = None,
    name: str = None,
): ...
def stream_recv_nb(
    ep: UCXEndpoint,
    buffer,
    nbytes: int,
    cb_func: Callable,
    cb_args: tuple = None,
    cb_kwargs: dict = None,
    name: str = None,
): ...
