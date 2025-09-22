""" Edge """

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from farms_network.core.options import EdgeOptions
from farms_network.models import EdgeTypes


cpdef enum types:

    #EDGE TYPES
    generic = GENERIC
    excitatory = EXCITATORY
    inhibitory = INHIBITORY
    cholinergic = CHOLINERGIC
    phase_coupling = PHASE_COUPLING


cdef class EdgeCy:
    """ Python interface to Edge C-Structure"""

    def __cinit__(self, edge_type: str, **kwargs):
        self._edge = <edge_t*>malloc(sizeof(edge_t))
        if self._edge is NULL:
            raise MemoryError("Failed to allocate memory for edge_t")
        self._edge.type = <int>types[edge_type]
        self._edge.params = NULL
        self._edge.nparams = 0

    def __dealloc__(self):
        if self._edge is not NULL:
            free(self._edge)

    def __init__(self, edge_type: str, **kwargs):
        ...

    @property
    def type(self):
        return self._edge.type

    @property
    def nparams(self):
        return self._edge.nparams
