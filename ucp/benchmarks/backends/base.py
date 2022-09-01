from abc import ABC, abstractmethod


class BaseServer(ABC):
    @abstractmethod
    def __init__(self, args, xp, queue):
        """
        Benchmark server.

        Parameters
        ----------
        args: argparse.Namespace
            Parsed command-line arguments that will be used as parameters during
            the `run` method.
        xp: module
            Module implementing the NumPy API to use for data generation.
        queue: Queue
            Queue object where server will put the port it is listening at.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run the benchmark server.

        The server is executed as follows:
        1. Start the listener and put port where it is listening into the queue
           registered in constructor;
        2. Setup any additional context (Active Message registration, memory buffers
           to reuse, etc.);
        3. Transfer data back-and-forth with client;
        4. Shutdown server.
        """
        pass


class BaseClient(ABC):
    @abstractmethod
    def __init__(self, args, xp, queue, server_address, port):
        """
        Benchmark client.

        Parameters
        ----------
        args: argparse.Namespace
            Parsed command-line arguments that will be used as parameters during
            the `run` method.
        xp: module
            Module implementing the NumPy API to use for data generation.
        queue: Queue
            Queue object where to put timing results.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run the benchmark client.

        The client is executed as follows:
        1. Connects to listener;
        2. Setup any additional context (Active Message registration, memory buffers
           to reuse, etc.);
        3. Transfer data back-and-forth with server;
        4. Shutdown client;
        5. Put timing results into the queue registered in constructor.
        """
        pass

    def print_backend_specific_config(self):
        """
        Pretty print configuration specific to backend implementation.
        """
        pass
