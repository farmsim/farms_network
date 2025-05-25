""" Node """

from typing import Optional

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from farms_network.core.options import NodeOptions
from farms_network.models import Models


cdef class NodeCy:
    """ Interface to Node C-Structure """

    MODEL = Models.BASE.value

    def __cinit__(self, **kwargs):
        self._node = <node_t*>malloc(sizeof(node_t))
        if self._node is NULL:
            raise MemoryError("Failed to allocate memory for node_t")
        self._node.nstates = 0
        self._node.ode = NULL
        self._node.output = NULL

        # Setup parameters
        self._node.params = NULL

    def __init__(self, **kwargs):
        ...

    def __dealloc__(self):
        if self._node is not NULL:
            free(self._node)

    # Property methods
    @property
    def nstates(self):
        return self._node.nstates

    @property
    def ninputs(self):
        return self._node.ninputs

    @property
    def nparams(self):
        return self._node.nparams

    @property
    def is_statefull(self):
        return self._node.is_statefull

    # Methods to wrap the ODE and output functions
    def ode(
            self,
            double time,
            double[:] states,
            double[:] derivatives,
            double external_input,
            double[:] network_outputs,
            unsigned int[:] inputs,
            double[:] weights,
            double noise,
    ):
        cdef double* states_ptr = &states[0]
        cdef double* derivatives_ptr = &derivatives[0]
        cdef double* network_outputs_ptr = &network_outputs[0]
        cdef unsigned int* inputs_ptr = &inputs[0]
        cdef double* weights_ptr = &weights[0]

        cdef edge_t** c_edges = NULL

        # Call the C function directly
        self._node.ode(
            time,
            states_ptr,
            derivatives_ptr,
            external_input,
            network_outputs_ptr,
            inputs_ptr,
            weights_ptr,
            noise,
            self._node,
            c_edges
        )

    def output(
            self,
            double time,
            double[:] states,
            double external_input,
            double[:] network_outputs,
            unsigned int[:] inputs,
            double[:] weights,
    ):
        # Call the C function and return its result
        cdef double* states_ptr = &states[0]
        cdef double* network_outputs_ptr = &network_outputs[0]
        cdef unsigned int* inputs_ptr = &inputs[0]
        cdef double* weights_ptr = &weights[0]
        cdef edge_t** c_edges = NULL
        return self._node.output(
            time,
            states_ptr,
            external_input,
            network_outputs_ptr,
            inputs_ptr,
            weights_ptr,
            self._node,
            c_edges
        )
