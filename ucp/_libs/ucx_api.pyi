import enum
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

# typedefs.pyx

class AllocatorType(enum.Enum):
    HOST: int
    CUDA: int
    UNSUPPORTED: int

class Feature(enum.Enum):
    TAG: int
    RMA: int
    AMO32: int
    AMO64: int
    WAKEUP: int
    STREAM: int
    AM: int

# utils.pyx

def get_current_options() -> Dict[str, str]: ...

def get_ucx_version() -> Tuple[int]: ...

def is_am_supported() -> bool: ...

# ucx_object.pyx

class UCXObject:
    def close(self) -> None: ...

# ucx_context.pyx

class UCXContext(UCXObject):
    def __init__(
        self, config_dict: Mapping = ..., feature_flags: Iterable[Feature] = ...
    ): ...

# ucx_address.pyx

class UCXAddress:
    @classmethod
    def from_buffer(cls, buffer) -> UCXAddress: ...
    @classmethod
    def from_worker(cls, worker: UCXWorker) -> UCXAddress: ...
    @property
    def address(self) -> int: ...
    @property
    def length(self) -> int: ...

# ucx_worker.pyx

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
    def register_am_allocator(
        self, allocator: Callable, allocator_type: AllocatorType
    ) -> None: ...

# ucx_listener.pyx

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

# ucx_endpoint.pyx

class UCXEndpoint(UCXObject):
    def info(self) -> str: ...
    @property
    def worker(self) -> UCXWorker: ...
    def unpack_rkey(self, rkey) -> UCXRkey: ...

# ucx_memory_handle.pyx

class UCXMemoryHandle(UCXObject):
    @classmethod
    def alloc(cls, ctx: UCXContext, size: int) -> UCXMemoryHandle: ...
    @classmethod
    def map(cls, ctx: UCXContext, buffer) -> UCXMemoryHandle: ...
    def pack_rkey(self) -> PackedRemoteKey: ...

# transfer.pyx

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

def am_send_nbx(
    ep: UCXEndpoint,
    buffer,
    nbytes: int,
    cb_func: Callable,
    cb_args: Optional[tuple] = ...,
    cb_kwargs: Optional[dict] = ...,
    name: Optional[str] = ...,
): ...

def am_recv_nb(
    ep: UCXEndpoint,
    cb_func: Callable,
    cb_args: Optional[tuple] = ...,
    cb_kwargs: Optional[dict] = ...,
    name: Optional[str] = ...,
): ...
