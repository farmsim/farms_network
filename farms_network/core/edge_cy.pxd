""" Edge Base Struture """


cdef enum:

    #EDGE TYPES
    GENERIC = 0
    EXCITATORY = 1
    INHIBITORY = 2
    CHOLINERGIC = 3
    PHASE_COUPLING = 4


cdef struct edge_t:
    unsigned int type           # Type of connection
    # Edge parameters
    unsigned int nparams
    void* params


cdef class EdgeCy:
    """ Python interface to Edge C-Structure """

    cdef:
        edge_t* _edge
