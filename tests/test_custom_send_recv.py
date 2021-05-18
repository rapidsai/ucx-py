import asyncio
import pickle

import numpy as np
import pytest

from distributed.utils import nbytes

import ucp

cudf = pytest.importorskip("cudf")
distributed = pytest.importorskip("distributed")
cuda = pytest.importorskip("numba.cuda")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        lambda cudf: cudf.Series([1, 2, 3], index=[4, 5, 6]),
        lambda cudf: cudf.Series([1, None, 3]),
        lambda cudf: cudf.Series(range(2 ** 13)),
        lambda cudf: cudf.DataFrame({"a": np.random.random(1200000)}),
        lambda cudf: cudf.DataFrame({"a": range(2 ** 20)}),
        lambda cudf: cudf.DataFrame({"a": range(2 ** 26)}),
        lambda cudf: cudf.Series(),
        lambda cudf: cudf.DataFrame(),
        lambda cudf: cudf.DataFrame({"a": [], "b": []}),
        lambda cudf: cudf.DataFrame({"a": [1.0], "b": [2.0]}),
        lambda cudf: cudf.DataFrame(
            {"a": ["a", "b", "c", "d"], "b": ["a", "b", "c", "d"]}
        ),
        lambda cudf: cudf.datasets.timeseries(),  # ts index with ints, cats, floats
    ],
)
async def test_send_recv_cudf(event_loop, g):
    # requires numba=0.45 (.nbytes)
    # or fix nbytes in distributed
    cudf = pytest.importorskip("cudf")

    class UCX:
        def __init__(self, ep):
            self.ep = ep

        async def write(self, cdf):
            header, _frames = cdf.serialize()
            frames = [pickle.dumps(header)] + _frames

            # Send meta data
            await self.ep.send(np.array([len(frames)], dtype=np.uint64))
            await self.ep.send(
                np.array(
                    [hasattr(f, "__cuda_array_interface__") for f in frames],
                    dtype=bool,
                )
            )
            await self.ep.send(np.array([nbytes(f) for f in frames], dtype=np.uint64))
            # Send frames
            for frame in frames:
                if nbytes(frame) > 0:
                    await self.ep.send(frame)

        async def read(self):
            try:
                # Recv meta data
                nframes = np.empty(1, dtype=np.uint64)
                await self.ep.recv(nframes)
                is_cudas = np.empty(nframes[0], dtype=bool)
                await self.ep.recv(is_cudas)
                sizes = np.empty(nframes[0], dtype=np.uint64)
                await self.ep.recv(sizes)
            except (ucp.exceptions.UCXCanceled, ucp.exceptions.UCXCloseError) as e:
                msg = "SOMETHING TERRIBLE HAS HAPPENED IN THE TEST"
                raise e(msg)
            else:
                # Recv frames
                frames = []
                for is_cuda, size in zip(is_cudas.tolist(), sizes.tolist()):
                    if size > 0:
                        if is_cuda:
                            frame = cuda.device_array((size,), dtype=np.uint8)
                        else:
                            frame = np.empty(size, dtype=np.uint8)
                        await self.ep.recv(frame)
                        frames.append(frame)
                    else:
                        if is_cuda:
                            frames.append(cuda.device_array((0,), dtype=np.uint8))
                        else:
                            frames.append(b"")
                return frames

    class UCXListener:
        def __init__(self):
            self.comm = None

        def start(self):
            async def serve_forever(ep):
                ucx = UCX(ep)
                self.comm = ucx

            self.ucp_server = ucp.create_listener(serve_forever)

    uu = UCXListener()
    uu.start()
    uu.address = ucp.get_address()
    uu.client = await ucp.create_endpoint(uu.address, uu.ucp_server.port)
    ucx = UCX(uu.client)
    await asyncio.sleep(0.2)
    msg = g(cudf)
    frames, _ = await asyncio.gather(uu.comm.read(), ucx.write(msg))
    ucx_header = pickle.loads(frames[0])
    cudf_buffer = frames[1:]
    typ = type(msg)
    res = typ.deserialize(ucx_header, cudf_buffer)

    from cudf.tests.utils import assert_eq

    assert_eq(res, msg)
    await uu.comm.ep.close()
    await uu.client.close()

    assert uu.client.closed()
    assert uu.comm.ep.closed()
    del uu.ucp_server
    ucp.reset()
