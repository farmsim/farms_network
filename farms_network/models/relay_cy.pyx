""" Relay model """

from libc.stdio cimport printf
from libc.stdlib cimport malloc


cdef double output(
    double time,
    double* states,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    node_t* c_node,
    edge_t** c_edges,
) noexcept:
    """ Node output. """
    return external_input


cdef class RelayNodeCy(NodeCy):
    """ Python interface to Relay Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.is_statefull = False
        self._node.output = output

    def __init__(self, **kwargs):
        super().__init__()

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')
