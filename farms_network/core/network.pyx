""" Neural Network """

from .neuron cimport Neuron
from libc.string cimport strdup


cdef class PyNetwork:
    """ Python interface to Network ODE """

    def __cinit__(self, nneurons: int):
        """ C initialization for manual memory allocation """

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        pass
