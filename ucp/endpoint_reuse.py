import ctypes
import uuid

import numpy as np

from . import core


class EPHandle:
    def __init__(self, ep):
        self.ep = ep
        self.refcount = 1


class EndpointReuse:
    """Class to seamlessly reuse endpoints.

    It uses the tag feature of send/recv to separate "virtual" endpoint
    pairs from each other.

    Warning
    -------
    When closing a reused endpoint, the peer might not be notified.

    Connection Protocol
    -------------------
    1) Client connect to server using a new endpoint.
    2) Client send the IDs of all its existing endpoints.
    3) Server receives the IDs from the client and checks if it has a
       matching existing endpoint. It then sends the matching ID to the
       client or zero if no match.
    4) The client and server now continue with either the existing matching
       endpoints or the new endpoints (which are registered for later reuse).
    """

    existing_endpoints = {}

    def __init__(self, handle, tag):
        self.handle = handle
        self.tag = tag

    @classmethod
    async def create_endpoint(cls, ip, port):
        tag = ctypes.c_uint32(uuid.uuid4().int).value
        ep_new = await core.create_endpoint(ip, port)

        existing_endpoints = list(cls.existing_endpoints.values())
        my_ep_ids = []
        for ep in existing_endpoints:
            if not ep.ep.closed():
                ep.refcount += 1
                my_ep_ids.append(ep.ep._msg_tag_recv)

        await ep_new.send(np.array([len(my_ep_ids), tag], dtype=np.uint64))
        await ep_new.send(np.array(my_ep_ids, dtype=np.uint64))

        reuse_ep_id = np.empty(1, dtype=np.uint64)
        await ep_new.recv(reuse_ep_id)
        reuse_ep_id = reuse_ep_id[0]

        for ep in existing_endpoints:
            if not ep.ep.closed():
                ep.refcount -= 1
                if ep.refcount == 0:
                    await ep.ep.close()

        if reuse_ep_id:
            reuse_ep = cls.existing_endpoints[reuse_ep_id]
            reuse_ep.refcount += 1
            await ep_new.close()
        else:
            reuse_ep = EPHandle(ep_new)
            assert ep_new._msg_tag_send not in cls.existing_endpoints
            cls.existing_endpoints[ep_new._msg_tag_send] = reuse_ep
        return cls(reuse_ep, tag)

    @classmethod
    def create_listener(cls, cb_coroutine, port):
        async def _handle(ep_new):
            msg = np.empty(2, dtype=np.uint64)
            await ep_new.recv(msg)
            n_peers_ep_ids, tag = (msg[0], np.uint32(msg[1]))
            peers_ep_ids = np.empty(n_peers_ep_ids, dtype=np.uint64)
            await ep_new.recv(peers_ep_ids)
            existing_ep = None
            for peers_ep_id in peers_ep_ids:
                existing_ep = cls.existing_endpoints.get(peers_ep_id)
                if existing_ep is not None and not existing_ep.ep.closed():
                    break

            if existing_ep:
                existing_ep.refcount += 1
                await ep_new.send(
                    np.array([existing_ep.ep._msg_tag_recv], dtype=np.uint64)
                )
                await ep_new.close()
            else:
                await ep_new.send(np.array([0], dtype=np.uint64))
                existing_ep = EPHandle(ep_new)
                assert ep_new._msg_tag_send not in cls.existing_endpoints
                cls.existing_endpoints[ep_new._msg_tag_send] = existing_ep

            await cb_coroutine(cls(existing_ep, tag))

        return core.create_listener(_handle, port=port)

    async def send(self, buffer, nbytes=None):
        await self.handle.ep.send(buffer, nbytes=nbytes, tag=self.tag)

    async def recv(self, buffer, nbytes=None):
        await self.handle.ep.recv(buffer, nbytes=nbytes, tag=self.tag)

    async def close(self):
        if self.closed():
            return
        self.handle.refcount -= 1
        if self.handle.refcount == 0:
            h = self.handle
            self.handle = None
            await h.ep.close()

    def closed(self):
        return self.handle is None or self.handle.ep.closed()

    def abort(self):
        if self.closed():
            return
        self.handle.refcount -= 1
        if self.handle.refcount == 0:
            self.handle.ep.abort()
            self.handle = None
