UCX-Py
======

.. warning::

    UCX-Py is deprecated, version 0.45 was its last release and it will receive no further updates. Users are advised to migrate to `UCXX <https://github.com/rapidsai/ucxx>`_ immediately. `UCXX documentation <https://docs.rapids.ai/api/ucxx/nightly/>`_.

UCX-Py is the Python interface for `UCX <https://github.com/openucx/ucx>`_, a low-level high-performance networking library.  UCX and UCX-Py supports several transport methods including InfiniBand and NVLink while still using traditional networking protocols like TCP.


.. image:: _static/Architecture.png
   :alt: A simple dask dictionary
   :align: center


.. toctree::
   :maxdepth: 1
   :hidden:

   quickstart
   install
   configuration
   deployment
   ucx-debug


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & reference

   os-limits
   transport-monitoring
   send-recv
   api
   glossary
