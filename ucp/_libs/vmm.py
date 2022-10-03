from functools import partial

from cuda import cuda

from dask_cuda.rmm_vmm_block_pool import VmmBlockPool
from dask_cuda.rmm_vmm_pool import checkCudaErrors
from dask_cuda.vmm_pool import VmmPool


def get_vmm_allocator(vmm):
    if vmm:
        vmm_is_block_pool = isinstance(vmm, VmmBlockPool)
        print(f"Server vmm_is_block_pool: {vmm_is_block_pool}")

        if isinstance(vmm, VmmBlockPool) or isinstance(vmm, VmmPool):
            vmm_allocator = VmmBlockPoolArray
        else:
            vmm_allocator = VmmSingleArray
        return partial(vmm_allocator, vmm)

    return None


class VmmSingleArray:
    def __init__(self, vmm_allocator, size):
        self.vmm_allocator = vmm_allocator

        self.ptr = cuda.CUdeviceptr(self.vmm_allocator.allocate(size))
        self.shape = (size,)

    def __del__(self):
        self.vmm_allocator.deallocate(int(self.ptr), self.shape[0])

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.shape),
            "typestr": "u1",
            "data": (self.ptr, False),
            "version": 2,
        }

    def copy_from_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_from_host: {hex(int(self.ptr))}", flush=True)
        print(f"copy_from_host: {type(arr)}")
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(self.ptr, arr.ctypes.data, self.shape[0], stream)
        )
        checkCudaErrors(cuda.cuStreamSynchronize(stream))

    def copy_to_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_to_host: {hex(int(self.ptr))}", flush=True)
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(arr.ctypes.data, self.ptr, self.shape[0], stream)
        )
        checkCudaErrors(cuda.cuStreamSynchronize(stream))


class VmmArraySlice:
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

    def copy_from_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_from_host: {hex(int(self.ptr))}", flush=True)
        print(f"copy_from_host: {type(arr)}")
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(self.ptr, arr.ctypes.data, self.shape[0], stream)
        )
        checkCudaErrors(cuda.cuStreamSynchronize(stream))

    def copy_to_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_to_host: {hex(int(self.ptr))}", flush=True)
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(arr.ctypes.data, self.ptr, self.shape[0], stream)
        )
        checkCudaErrors(cuda.cuStreamSynchronize(stream))


class VmmBlockPoolArray:
    def __init__(self, vmm_block_pool_allocator, size):
        self.vmm_allocator = vmm_block_pool_allocator

        self.ptr = cuda.CUdeviceptr(self.vmm_allocator.allocate(size))
        self.shape = (size,)

    def __del__(self):
        try:
            self.vmm_allocator.deallocate(int(self.ptr), self.shape[0])
        except Exception:
            print("EXCEPTION")

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.shape),
            "typestr": "u1",
            "data": (int(self.ptr), False),
            "version": 2,
        }

    def get_blocks(self):
        if isinstance(self.vmm_allocator, VmmBlockPool):
            blocks = self.vmm_allocator.get_allocation_blocks(int(self.ptr))
            return list([VmmArraySlice(block[0], block[1]) for block in blocks])
        else:
            blocks = self.vmm_allocator._allocs[int(self.ptr)].blocks
            return list([VmmArraySlice(block._ptr, block.size) for block in blocks])

    def copy_from_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_from_host: {hex(int(self.ptr))}", flush=True)
        print(f"copy_from_host: {type(arr)}")
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(self.ptr, arr.ctypes.data, self.shape[0], stream)
        )
        checkCudaErrors(cuda.cuStreamSynchronize(stream))

    def copy_to_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_to_host: {hex(int(self.ptr))}", flush=True)
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(arr.ctypes.data, self.ptr, self.shape[0], stream)
        )
        checkCudaErrors(cuda.cuStreamSynchronize(stream))
