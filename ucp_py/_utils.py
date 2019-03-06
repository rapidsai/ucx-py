import fcntl
import socket
import struct


def get_address(ifname='ib0'):
    """
    Get the address associated with a network interface.

    Parameters
    ----------
    ifname : str, default ib0
        The network interface name to find the address for.
        By default, 'ib0' is used. An OSError is raised for
        invalid interfaces.

    Returns
    -------
    address : str
        The inet addr associated with an interface.

    Examples
    --------
    >>> get_address()
    '10.33.225.160'

    >>> get_address(ifname='lo')
    '127.0.0.1'
    """
    # https://stackoverflow.com/a/24196955/1889400
    ifname = ifname.encode()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])


def make_server(cuda_info=None):
    async def echo_server(ep, lf):
        """
        Basic echo server for sized messages.

        We expect the other endpoint to follow the pattern::

        >>> await ep.send_obj(msg_size)  # size of the real message
        >>> await ep.send_obj(obj)       # send the real message
        >>> await ep.recv_obj(msg_size)  # receive the echo
        """
        from ucp_py._libs.ucp_py import destroy_ep, stop_listener

        while True:
            size_msg = await ep.recv_future()
            size = int(size_msg.get_obj())

            if not size:
                break

            msg = await ep.recv_obj(size, cuda=bool(cuda_info))
            obj = msg.get_obj()

            if cuda_info:
                import cupy
                if 'typestr' in cuda_info:
                    obj.typestr = cuda_info['typestr']
                obj = cupy.asarray(obj)

            await ep.send_obj(obj)

        destroy_ep(ep)
        stop_listener(lf)

    return echo_server
