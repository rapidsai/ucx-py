# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import weakref


def _handle_finalizer_wrapper(
    children, handle_finalizer, handle_as_int, *extra_args, **extra_kargs
):
    for weakref_to_child in children:
        child = weakref_to_child()
        if child is not None:
            child.close()
    handle_finalizer(handle_as_int, *extra_args, **extra_kargs)


cdef class UCXObject:
    """Base class for UCX classes

    This base class streamlines the cleanup of UCX objects and reduces duplicate code.
    """
    cdef:
        object __weakref__
        object _finalizer
        list _children

    def __cinit__(self):
        # The finalizer, which can be called multiple times but only
        # evoke the finalizer function once.
        # Is None when the underlying UCX handle hasen't been initialized.
        self._finalizer = None
        # List of weak references of UCX objects that make use of this object
        self._children = []

    cpdef void close(self) except *:
        """Close the object and free the underlying UCX handle.
        Does nothing if the object is already closed
        """
        if self.initialized:
            self._finalizer()

    @property
    def initialized(self):
        """Is the underlying UCX handle initialized"""
        return self._finalizer and self._finalizer.alive

    cpdef void add_child(self, child) except *:
        """Add a UCX object to this object's children. The underlying UCX
        handle will be freed when this obejct is freed.
        """
        self._children.append(weakref.ref(child))

    def add_handle_finalizer(self, handle_finalizer, handle_as_int, *extra_args):
        """Add a finalizer of `handle_as_int`"""
        self._finalizer = weakref.finalize(
            self,
            _handle_finalizer_wrapper,
            self._children,
            handle_finalizer,
            handle_as_int,
            *extra_args
        )
