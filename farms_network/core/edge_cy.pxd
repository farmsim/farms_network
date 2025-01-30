""" Edge Base Struture """


cdef enum:

    #EDGE TYPES
    OPEN = 0
    EXCITATORY = 1
    INHIBITORY = 2
    CHOLINERGIC = 3


cdef struct edge_t:

    #
    char* source                # Source node
    char* target                # Target node
    unsigned int type           # Type of connection
    char* model                 # Type of the model (e.g., "base")

    # Edge parameters
    unsigned int nparameters
    void* parameters



cdef class EdgeCy:
    """ Python interface to Edge C-Structure"""

    cdef:
        edge_t* _edge
