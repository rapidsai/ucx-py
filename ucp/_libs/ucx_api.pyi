import enum
from typing import Callable, Iterable, Mapping, Optional

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
        cb_args: Optional[tuple] = ...,
        cb_kwargs: dict = ...,
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
    cb_args: Optional[tuple] = ...,
    cb_kwargs: Optional[dict] = ...,
    name: Optional[str] = ...,
): ...
def tag_recv_nb(
    worker: UCXWorker,
    buffer,
    nbytes: int,
    tag: int,
    cb_func: Callable,
    cb_args: Optional[tuple] = ...,
    cb_kwargs: Optional[dict] = ...,
    name: Optional[str] = ...,
    ep: Optional[UCXEndpoint] = ...,
): ...
def stream_send_nb(
    ep: UCXEndpoint,
    buffer,
    nbytes: int,
    cb_func: Callable,
    cb_args: Optional[tuple] = ...,
    cb_kwargs: Optional[dict] = ...,
    name: Optional[str] = ...,
): ...
def stream_recv_nb(
    ep: UCXEndpoint,
    buffer,
    nbytes: int,
    cb_func: Callable,
    cb_args: Optional[tuple] = ...,
    cb_kwargs: Optional[dict] = ...,
    name: Optional[str] = ...,
): ...
