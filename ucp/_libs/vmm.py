from functools import partial
from typing import Union

import numpy as np
from cuda import cuda

from dask_cuda.rmm_vmm_block_pool import VmmBlockPool
from dask_cuda.rmm_vmm_pool import checkCudaErrors
from dask_cuda.vmm_pool import VmmPool


def get_vmm_allocator(vmm):
    if vmm:
        if isinstance(vmm, VmmBlockPool) or isinstance(vmm, VmmPool):
            vmm_allocator = VmmBlockPoolArray
        else:
            vmm_allocator = VmmSingleArray
        return partial(vmm_allocator, vmm)

    return None


def copy_to_host(
    dst: np.ndarray,
    src: Union[int, cuda.CUdeviceptr],
    size: int,
    stream: cuda.CUstream = cuda.CUstream(0),
):
    if isinstance(src, int):
        src = cuda.CUdeviceptr(src)
    assert isinstance(src, cuda.CUdeviceptr)
    assert isinstance(dst, np.ndarray)
    assert isinstance(size, int)
    assert size > 0
    # print(
    #     f"copy_to_host src: {hex(int(src))}, dst: {hex(int(dst.ctypes.data))}",
    #     flush=True
    # )
    checkCudaErrors(cuda.cuMemcpyDtoHAsync(dst.ctypes.data, src, size, stream))
    checkCudaErrors(cuda.cuStreamSynchronize(stream))


def copy_to_device(
    dst: Union[int, cuda.CUdeviceptr],
    src: np.ndarray,
    size: int,
    stream: cuda.CUstream = cuda.CUstream(0),
):
    assert isinstance(src, np.ndarray)
    if isinstance(dst, int):
        dst = cuda.CUdeviceptr(dst)
    assert isinstance(dst, cuda.CUdeviceptr)
    assert isinstance(size, int)
    assert size > 0
    # print(
    #     f"copy_to_device src: {hex(int(src.ctypes.data))}, dst: {hex(int(dst))}",
    #     flush=True
    # )
    checkCudaErrors(cuda.cuMemcpyHtoDAsync(dst, src.ctypes.data, size, stream))
    checkCudaErrors(cuda.cuStreamSynchronize(stream))


class VmmAllocBase:
    def __init__(self, ptr, size):
        self.ptr = cuda.CUdeviceptr(ptr)
        self.shape = (size,)

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.shape),
            "typestr": "u1",
            "data": (int(self.ptr), False),
            "version": 2,
        }

    @property
    def nbytes(self):
        return self.shape[0]


class VmmArraySlice(VmmAllocBase):
    pass


class VmmSingleArray(VmmAllocBase):
    def __init__(self, vmm_allocator, size):
        ptr = cuda.CUdeviceptr(vmm_allocator.allocate(size))
        super().__init__(ptr, size)

        self.vmm_allocator = vmm_allocator

    def __del__(self):
        self.vmm_allocator.deallocate(int(self.ptr), self.shape[0])


class VmmBlockPoolArray(VmmAllocBase):
    def __init__(self, vmm_block_pool_allocator, size):
        ptr = cuda.CUdeviceptr(vmm_block_pool_allocator.allocate(size))
        super().__init__(ptr, size)

        self.vmm_allocator = vmm_block_pool_allocator

    def __del__(self):
        self.vmm_allocator.deallocate(int(self.ptr), self.shape[0])

    def get_blocks(self):
        if isinstance(self.vmm_allocator, VmmBlockPool):
            blocks = self.vmm_allocator.get_allocation_blocks(int(self.ptr))
            return list([VmmArraySlice(block[0], block[1]) for block in blocks])
        else:
            blocks = self.vmm_allocator._allocs[int(self.ptr)].blocks
            return list([VmmArraySlice(block._ptr, block.size) for block in blocks])
