"""
Benchmark send receive on one machine
"""
import argparse
import asyncio
import cProfile
import io
import pickle
import pstats
import sys
from time import perf_counter as clock

import cupy
import numpy as np

from dask.utils import format_bytes, format_time

import cudf
import rmm

import ucp
from ucp.utils import run_on_local_network


async def send_df(ep, df):
    header, frames = df.serialize()
    header["frame_ifaces"] = [f.__cuda_array_interface__ for f in frames]
    header = pickle.dumps(header)
    header_nbytes = np.array([len(header)], dtype=np.uint64)
    await ep.send(header_nbytes)
    await ep.send(header)
    for frame in frames:
        await ep.send(frame)


async def recv_df(ep):
    header_nbytes = np.empty((1,), dtype=np.uint64)
    await ep.recv(header_nbytes)
    header = bytearray(header_nbytes[0])
    await ep.recv(header)
    header = pickle.loads(header)

    frames = [
        cupy.empty(iface["shape"], dtype=iface["typestr"])
        for iface in header["frame_ifaces"]
    ]
    for frame in frames:
        await ep.recv(frame)

    cudf_typ = pickle.loads(header["type-serialized"])
    return cudf_typ.deserialize(header, frames)


async def barrier(rank, eps):
    if rank == 0:
        await asyncio.gather(*[ep.recv(np.empty(1, dtype="u1")) for ep in eps.values()])
    else:
        await eps[0].send(np.zeros(1, dtype="u1"))


async def send_bins(eps, bins):
    futures = []
    for rank, ep in eps.items():
        futures.append(send_df(ep, bins[rank]))
    await asyncio.gather(*futures)


async def recv_bins(eps, bins):
    futures = []
    for ep in eps.values():
        futures.append(recv_df(ep))
    bins.extend(await asyncio.gather(*futures))


async def exchange_and_concat_bins(rank, eps, bins, timings=None):
    ret = [bins[rank]]
    if timings is not None:
        t1 = clock()
    await asyncio.gather(recv_bins(eps, ret), send_bins(eps, bins))
    if timings is not None:
        t2 = clock()
        timings.append(
            (t2 - t1, sum([sys.getsizeof(b) for i, b in enumerate(bins) if i != rank]))
        )
    return cudf.concat(ret)


async def distributed_join(args, rank, eps, left_table, right_table, timings=None):
    left_bins = left_table.partition_by_hash(["key"], args.n_chunks)
    right_bins = right_table.partition_by_hash(["key"], args.n_chunks)

    left_df = await exchange_and_concat_bins(rank, eps, left_bins, timings)
    right_df = await exchange_and_concat_bins(rank, eps, right_bins, timings)
    return left_df.merge(right_df)


def generate_chunk(i_chunk, local_size, num_chunks, chunk_type, frac_match):
    cupy.random.seed(42)

    if chunk_type == "build":
        # Build dataframe
        #
        # "key" column is a unique sample within [0, local_size * num_chunks)
        #
        # "shuffle" column is a random selection of partitions (used for shuffle)
        #
        # "payload" column is a random permutation of the chunk_size

        start = local_size * i_chunk
        stop = start + local_size

        df = cudf.DataFrame(
            {
                "key": cupy.arange(start, stop=stop, dtype="int64"),
                "payload": cupy.arange(local_size, dtype="int64"),
            }
        )
    else:
        # Other dataframe
        #
        # "key" column matches values from the build dataframe
        # for a fraction (`frac_match`) of the entries. The matching
        # entries are perfectly balanced across each partition of the
        # "base" dataframe.
        #
        # "payload" column is a random permutation of the chunk_size

        # Step 1. Choose values that DO match
        sub_local_size = local_size // num_chunks
        sub_local_size_use = max(int(sub_local_size * frac_match), 1)
        arrays = []
        for i in range(num_chunks):
            bgn = (local_size * i) + (sub_local_size * i_chunk)
            end = bgn + sub_local_size
            ar = cupy.arange(bgn, stop=end, dtype="int64")
            arrays.append(cupy.random.permutation(ar)[:sub_local_size_use])
        key_array_match = cupy.concatenate(tuple(arrays), axis=0)

        # Step 2. Add values that DON'T match
        missing_size = local_size - key_array_match.shape[0]
        start = local_size * num_chunks + local_size * i_chunk
        stop = start + missing_size
        key_array_no_match = cupy.arange(start, stop=stop, dtype="int64")

        # Step 3. Combine and create the final dataframe chunk
        key_array_combine = cupy.concatenate(
            (key_array_match, key_array_no_match), axis=0
        )
        df = cudf.DataFrame(
            {
                "key": cupy.random.permutation(key_array_combine),
                "payload": cupy.arange(local_size, dtype="int64"),
            }
        )
    return df


async def worker(rank, eps, args):
    # Setting current device and make RMM use it
    dev_id = args.devs[rank % len(args.devs)]
    cupy.cuda.runtime.setDevice(dev_id)
    rmm.reinitialize(
        pool_allocator=True, devices=dev_id, initial_pool_size=args.rmm_init_pool_size
    )

    # Make cupy use RMM
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    df1 = generate_chunk(rank, args.chunk_size, args.n_chunks, "build", args.frac_match)
    df2 = generate_chunk(rank, args.chunk_size, args.n_chunks, "other", args.frac_match)

    # Let's warmup and sync before benchmarking
    await distributed_join(args, rank, eps, df1, df2)
    await barrier(rank, eps)

    if args.cuda_profile:
        cupy.cuda.profiler.start()

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    timings = []
    t1 = clock()
    await distributed_join(args, rank, eps, df1, df2, timings)
    await barrier(rank, eps)
    took = clock() - t1

    if args.profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.dump_stats("%s.%0d" % (args.profile, rank))

    if args.cuda_profile:
        cupy.cuda.profiler.stop()

    data_processed = len(df1) * sum([t.itemsize for t in df1.dtypes])
    data_processed += len(df2) * sum([t.itemsize for t in df2.dtypes])

    return {
        "bw": sum(t[1] for t in timings) / sum(t[0] for t in timings),
        "wallclock": took,
        "throughput": args.n_chunks * data_processed / took,
        "data_processed": data_processed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--chunks-per-dev",
        metavar="N",
        default=1,
        type=int,
        help="Number of chunks per device",
    )
    parser.add_argument(
        "-d",
        "--devs",
        metavar="LIST",
        default="0",
        type=str,
        help='GPU devices to use (default "0").',
    )
    parser.add_argument(
        "-s",
        "--server-address",
        metavar="ip",
        default=ucp.get_address(),
        type=str,
        help="Server address (default `ucp.get_address()`).",
    )
    parser.add_argument("-c", "--chunk-size", type=int, default=4, metavar="N")
    parser.add_argument(
        "--frac-match",
        metavar="FRAC",
        default=0.3,
        type=float,
        help="Fraction of rows that matches (default 0.3)",
    )
    parser.add_argument(
        "--net-devices",
        metavar="LIST",
        default=None,
        type=str,
        help='List of net devices to use, one for each device or "auto"',
    )
    parser.add_argument(
        "--profile",
        metavar="FILENAME",
        default=None,
        type=str,
        help="Write profile for each worker to `filename.RANK`",
    )
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Enable CUDA profiling, use with `nvprof --profile-child-processes \
                --profile-from-start off`",
    )
    parser.add_argument(
        "--rmm-init-pool-size",
        metavar="BYTES",
        default=None,
        type=int,
        help="Initial RMM pool size (default  1/2 total GPU memory)",
    )
    args = parser.parse_args()
    args.devs = [int(d) for d in args.devs.split(",")]
    args.n_chunks = len(args.devs) * args.chunks_per_dev
    if args.n_chunks < 2:
        raise RuntimeError(
            f"Number of chunks must be greater than 1 (chunks-per-dev: \
                    {args.chunks_per_dev}, devs: {args.devs})"
        )
    if args.net_devices == "auto":
        args.net_devices = [ucp.utils.get_closest_net_devices(d) for d in args.devs]
    elif args.net_devices is not None:
        args.net_devices = args.net_devices.split(",")
        assert len(args.net_devices) == len(args.devs)
    return args


def main():
    args = parse_args()
    ranks = range(args.n_chunks)
    assert len(ranks) > 1
    assert len(ranks) % 2 == 0

    ucx_options_list = None
    if args.net_devices is not None:
        ucx_options_list = [
            {"NET_DEVICES": args.net_devices[rank % len(args.devs)]} for rank in ranks
        ]

    stats = run_on_local_network(
        args.n_chunks,
        worker,
        worker_args=args,
        server_address=args.server_address,
        ucx_options_list=ucx_options_list,
    )

    wc = stats[0]["wallclock"]
    bw = sum(s["bw"] for s in stats) / len(stats)
    tp = stats[0]["throughput"]
    dp = sum(s["data_processed"] for s in stats)

    print("cudf merge benchmark")
    print("----------------------------")
    print(f"device(s)      | {args.devs}")
    print(f"chunks-per-dev | {args.chunks_per_dev}")
    print(f"rows-per-chunk | {args.chunk_size}")
    print(f"data-processed | {format_bytes(dp)}")
    print(f"frac-match     | {args.frac_match}")
    print("============================")
    print(f"Wall-clock     | {format_time(wc)}")
    print(f"Bandwidth      | {format_bytes(bw)}/s")
    print(f"Throughput     | {format_bytes(tp)}/s")


if __name__ == "__main__":
    main()
