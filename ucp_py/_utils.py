import fcntl
import socket
import struct
import sys


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
    """
    # https://stackoverflow.com/a/24196955/1889400
    ifname = ifname.encode()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])


def sizeof(obj):
    """
    Get the size of an object, defined as

    1. it's length
    2. it's Python overhead

    So this is appropriate for sequences where each element
    is a byte (bytestrings or memoryviews).
    """
    return len(obj) + sys.getsizeof(obj[:0])
