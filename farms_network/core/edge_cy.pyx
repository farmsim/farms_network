""" Edge """

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from farms_network.core.options import EdgeOptions


cpdef enum types:

    #EDGE TYPES
    open = OPEN
    excitatory = EXCITATORY
    inhibitory = INHIBITORY
    cholinergic = CHOLINERGIC


cdef class EdgeCy:
    """ Python interface to Edge C-Structure"""

    def __cinit__(self, source: str, target: str, edge_type: str, model: str):
        self._edge = <edge_t*>malloc(sizeof(edge_t))
        if self._edge is NULL:
            raise MemoryError("Failed to allocate memory for edge_t")
        self._edge.source = strdup(source.encode('UTF-8'))
        self._edge.target = strdup(target.encode('UTF-8'))
        self._edge.type = <int>types[edge_type]
        if model is None:
            model = ""
        self._edge.model = strdup(model.encode('UTF-8'))
        self._edge.parameters = NULL
        self._edge.nparameters = 0

    def __dealloc__(self):
        if self._edge is not NULL:
            if self._edge.source is not NULL:
                free(self._edge.source)
            if self._edge.target is not NULL:
                free(self._edge.target)
            if self._edge.parameters is not NULL:
                free(self._edge.parameters)
            free(self._edge)

    def __init__(self, source: str, target: str, edge_type: str, model: str):
        ...

    @property
    def source(self):
        if self._edge.source is NULL:
            return None
        return self._edge.source.decode('UTF-8')

    @property
    def target(self):
        if self._edge.target is NULL:
            return None
        return self._edge.target.decode('UTF-8')

    @property
    def type(self):
        return self._edge.type

    @property
    def nparameters(self):
        return self._edge.nparameters

    @property
    def parameters(self):
        """Generic accessor for parameters."""
        if not self._edge.parameters:
            raise ValueError("edge_t parameters are NULL")
        if self._edge.nparameters == 0:
            raise ValueError("No parameters available")

        # The derived class should override this method to provide specific behavior
        raise NotImplementedError("Base class does not define parameter handling")
