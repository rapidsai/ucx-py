# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio
import socket
import weakref


class ProgressTask(object):
    def __init__(self, worker, event_loop):
        """Creates a task that keeps calling worker.progress()

        Notice, class and created task is carefull not to hold a
        reference to `worker` so that a danling progress task will
        not prevent `worker` to be garbage collected.

        Parameters
        ----------
        worker: UCXWorker
            The UCX worker context to progress
        event_loop: asyncio.EventLoop
            The event loop to do progress in.
        """
        self.weakref_worker = weakref.ref(worker)
        self.event_loop = event_loop
        self.asyncio_task = None

    def __del__(self):
        if self.asyncio_task is not None:
            self.asyncio_task.cancel()

    # Hash and equality is based on the event loop
    def __hash__(self):
        return hash(self.event_loop)

    def __eq__(self, other):
        return hash(self) == hash(other)


class NonBlockingMode(ProgressTask):
    def __init__(self, worker, event_loop):
        super().__init__(worker, event_loop)
        self.asyncio_task = event_loop.create_task(self._progress_task())

    async def _progress_task(self):
        """This helper function maintains a UCX progress loop."""
        while True:
            worker = self.weakref_worker()
            if worker is None or not worker.initialized:
                return
            worker.progress()
            del worker
            # Give other co-routines a chance to run.
            await asyncio.sleep(0)


class BlockingMode(ProgressTask):
    def __init__(self, worker, event_loop, epoll_fd):
        super().__init__(worker, event_loop)

        # Creating a job that is ready straightaway but with low priority.
        # Calling `await self.event_loop.sock_recv(self.rsock, 1)` will
        # return when all non-IO tasks are finished.
        # See <https://stackoverflow.com/a/48491563>.
        self.rsock, wsock = socket.socketpair()
        self.rsock.setblocking(0)
        wsock.setblocking(0)
        wsock.close()

        # Bind an asyncio reader to a UCX epoll file descripter
        event_loop.add_reader(epoll_fd, self._fd_reader_callback)

        # Remove the reader and close socket on finalization
        weakref.finalize(self, event_loop.remove_reader, epoll_fd)
        weakref.finalize(self, self.rsock.close)

    def _fd_reader_callback(self):
        worker = self.weakref_worker()
        if worker is None or not worker.initialized:
            return
        worker.progress()

        # Notice, we can safely overwrite `self.dangling_arm_task`
        # since previous arm task is finished by now.
        assert self.asyncio_task is None or self.asyncio_task.done()
        self.asyncio_task = self.event_loop.create_task(self._arm_worker())

    async def _arm_worker(self):
        # When arming the worker, the following must be true:
        #  - No more progress in UCX (see doc of ucp_worker_arm())
        #  - All asyncio tasks that isn't waiting on UCX must be executed
        #    so that the asyncio's next state is epoll wait.
        #    See <https://github.com/rapidsai/ucx-py/issues/413>
        while True:
            worker = self.weakref_worker()
            if worker is None or not worker.initialized:
                return
            worker.progress()

            # Cancel inflight messages that couldn't be completed. This may
            # happen if the user called ep.recv() but the remote worker
            # errored before sending the message.
            if worker.cancel_inflight_messages() > 0:
                worker.progress()

            del worker

            # This IO task returns when all non-IO tasks are finished.
            # Notice, we do NOT hold a reference to `worker` while waiting.
            await self.event_loop.sock_recv(self.rsock, 1)

            worker = self.weakref_worker()
            if worker is None or not worker.initialized:
                return
            if worker.arm():
                # At this point we know that asyncio's next state is
                # epoll wait.
                break
