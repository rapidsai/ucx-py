API
===

.. currentmodule:: ucp

**ucp**

.. autosummary::
   ucp
   ucp.create_listener
   ucp.create_endpoint
   ucp.get_address
   ucp.get_config
   ucp.get_ucp_worker
   ucp.get_ucx_version
   ucp.init
   ucp.progress
   ucp.reset

**Endpoint**

.. autosummary::
   Endpoint
   Endpoint.abort
   Endpoint.close
   Endpoint.closed
   Endpoint.close_after_n_recv
   Endpoint.cuda_support
   Endpoint.get_ucp_endpoint
   Endpoint.get_ucp_worker
   Endpoint.recv
   Endpoint.send
   Endpoint.ucx_info
   Endpoint.uid

**Listener**

.. autosummary::
   Listener
   Listener.close
   Listener.closed
   Listener.port

.. currentmodule:: ucp

.. autofunction:: create_listener
.. autofunction:: create_endpoint
.. autofunction:: get_address
.. autofunction:: get_config
.. autofunction:: get_ucp_worker
.. autofunction:: get_ucx_version
.. autofunction:: init
.. autofunction:: progress
.. autofunction:: reset

Endpoint
--------

.. currentmodule:: ucp

.. autoclass:: Endpoint
   :members:


Listener
--------

.. currentmodule:: ucp

.. autoclass:: Listener
   :members:
