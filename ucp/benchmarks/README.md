# Running Benchmarks

## cuDF Merge

The cuDF merge benchmark can be executed in 3 different ways:

1. Local-node only;
1. Multi-node with automatic process launch via SSH;
1. Multi-node with manual process launch.

In the following subsections we will exemplify how to launch the benchmark for each case. Be sure to check `python -m ucp.benchmarks.cudf_merge --help` to see a complete description of available options and their description.


### Local-node only

This is the simplest setup and will execute the benchmark only on GPUs where the process is being launched from. In its simple form, it can be executed as:

```bash
python -m ucp.benchmarks.cudf_merge -d 0,1 -c 1_000_000 --iter 10
```

The process above will execute the benchmark with the first 2 GPUs in the node, with 1M rows per chunk and a single chunk per node, for 10 iterations. To extend the same to 8 GPUs we could simply write:

```bash
python -m ucp.benchmarks.cudf_merge -d 0,1,2,3,4,5,6,7 -c 1_000_000 --iter 10
```


### Multi-node with automatic process launch via SSH

In this setup, the user can run the benchmark spanning multiple nodes, but a password-less SSH setup is required. That is often achieved by having a private key that doesn't require a passphrase authorized to login to all the nodes where the benchmark is expected to run. To test if this setup is already correct, it should suffice to execute `ssh NODE_HOSTNAME` or `ssh NODE_IP`, if you get a shell in the remote machine without being asked for a password, you are probably good to go. If this is not yet setup, please consult the documentation of your operating systems to set this up.

Once the SSH setup is complete, we can extend the local-node example by simply introducing a new argument `--hosts`. The `--hosts` argument takes a comma-separated list of node hostnames or IP addresses, where the first element is where the server will run, followed by any number of worker hosts. The server is required to synchronize workers and doesn't require a GPU but is allowed to share the same node as one of the worker nodes in that list. For example, if we want the server to run on a node with hostname 'dgx12', with workers on machine with hostname 'dgx12', another with IP address '10.10.10.10' and another with hostname 'dgx13', we would run the command as below:

```bash
python -m ucp.benchmarks.cudf_merge -d 0,1,2,3 -c 1_000_000 --iter 10 --hosts dgx12,dgx12,10.10.10.10,dgx15
```

Note that in the example above we repeated dgx12, the first node identifies the server, and the remaining arguments identify workers. Multi-node setups also require every node to use the same GPU indices, so if you specify `-d 0,1,2,3`, all worker nodes must have those four GPUs available.

Should anything go wrong, you may specify the `UCXPY_ASYNCSSH_LOG_LEVEL=DEBUG` environment variable. This will print additional information, including the exact command that is being executed on the server and workers, as well as the output for each process.


### Multi-node with manual process setup

This is by far the most complicated setup, and should be used only if running on a cluster where SSH isn't possible or not recommended.

Before diving in to the details of this setup, the user should know there is a convenience function to generate commands for each of the nodes. To generate that, the user may refer to the SSH section, where we pass `--hosts` to specify server and worker nodes, this time with an added `--print-commands-only` argument. The output of the command will print one line for server and each node with the exact command that should be executed on those nodes.

Now if the user would like to continue and learn the details this setup, the user is required specify the exact amount of workers that will be executed by the cluster, and the index of each node. Unlike the SSH setup, this time we may not use `--hosts` to specify where to launch.

**WARNING: All processes must specify the same values for `--devs`, `--chunks-per-dev`, `--chunk-size`, `--frac-match`, `--iter`, `--warmup-iter`, `--num-workers`.**


#### Calculating number of workers

Calculating the number of workers is straightforward but critical to be done right. It can be calculated as follows:

```python
# num_workers: number of workers, passed as `--num-workers`
# len(devs): length of the GPU list specified via `--devs`
# chunks_per_dev: number of chunks (processes) per GPU specified via `--chunks-per-dev`
# num_worker_nodes: number of nodes that will run workers
num_workers = len(devs) * chunks_per_dev * num_worker_nodes
```

In the examples that follow, we will launch 4 GPUs per node, 2 chunks per device and 2 worker nodes.

```python
# num_workers = len(devs) * chunks_per_dev * num_worker_nodes
16 = 4 * 2 * 2
```

#### Server

First, the user must launch the server, which may be in a node that doesn't contain any GPUs, or could be on one of the nodes where workers will run. To do so, two options exist:

##### Address on a file

In this setup, a file that is reachable on all nodes of the cluster must be specified, for example in a network file system. 

```bash
python -m ucp.benchmarks.cudf_merge --server --devs 0,1,2,3 --chunks-per-dev 2 --chunk-size 1000000 --frac-match 0.5 --iter 10 --warmup-iter 5 --num-workers 16 --server-file /path/to/network/fs/server.json
```

##### Address on stdout

```bash
python -m ucp.benchmarks.cudf_merge --server --devs 0,1,2,3 --chunks-per-dev 2 --chunk-size 1000000 --frac-match 0.5 --iter 10 --warmup-iter 5 --num-workers 16
```


#### Workers

Once the server is up and running, workers must be launched on multiple nodes. Each node must execute workers with all options matching other nodes, both workers and server, with the execption of `--node-idx`. The `--node-idx` argument is used as a unique identifier to each worker, so that it may compute the rank for each GPU worker.


#### Server address on a file

```bash
# Worker 0
python -m ucp.benchmarks.cudf_merge --devs 0,1,2,3 --chunks-per-dev 2 --chunk-size 1000000 --frac-match 0.5 --iter 10 --warmup-iter 5 --num-workers 16 --node-idx 0 --rmm-init-pool-size 4000000000 --server-file '/path/to/network/fs/server.json'

# Worker 1
python -m ucp.benchmarks.cudf_merge --devs 0,1,2,3 --chunks-per-dev 2 --chunk-size 1000000 --frac-match 0.5 --iter 10 --warmup-iter 5 --num-workers 16 --node-idx 1 --rmm-init-pool-size 4000000000 --server-file '/path/to/network/fs/server.json'
```


#### Server address on stdout

```bash
# Worker 0
python -m ucp.benchmarks.cudf_merge --devs 0,1,2,3 --chunks-per-dev 2 --chunk-size 1000000 --frac-match 0.5 --iter 10 --warmup-iter 5 --num-workers 16 --node-idx 0 --rmm-init-pool-size 4000000000 --server-address 'REPLACE WITH SERVER ADDRESS'

# Worker 1
python -m ucp.benchmarks.cudf_merge --devs 0,1,2,3 --chunks-per-dev 2 --chunk-size 1000000 --frac-match 0.5 --iter 10 --warmup-iter 5 --num-workers 16 --node-idx 1 --rmm-init-pool-size 4000000000 --server-address 'REPLACE WITH SERVER ADDRESS'
```
