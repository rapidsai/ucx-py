import fcntl
import socket
import struct
import os


def get_address(ifname=None):
    """
    Get the address associated with a network interface.

    Parameters
    ----------
    ifname : str
        The network interface name to find the address for.
        If None, it uses the value of environment variable `UCXPY_IFNAME`
        and if `UCXPY_IFNAME` is not set it defaults to "ib0"
        An OSError is raised for invalid interfaces.

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
    if ifname is None:
        ifname = os.environ.get("UCXPY_IFNAME", "ib0")

    ifname = ifname.encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(), 0x8915, struct.pack("256s", ifname[:15])  # SIOCGIFADDR
        )[20:24]
    )
