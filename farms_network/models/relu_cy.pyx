""" Rectified Linear Unit """


from libc.stdio cimport printf
from libc.stdlib cimport free, malloc


cpdef enum STATE:

    #STATES
    nstates = NSTATES


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
    cdef relu_params_t params = (<relu_params_t*> c_node[0].params)[0]

    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight
    cdef unsigned int ninputs = c_node.ninputs
    _sum += external_input
    for j in range(ninputs):
        _input = network_outputs[inputs[j]]
        _weight = weights[j]
        _sum += _weight*_input

    cdef double res = max(0.0, params.gain*(params.sign*_sum + params.offset))
    return res


cdef class ReLUNodeCy(NodeCy):
    """ Python interface to ReLU Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.nstates = 0
        self._node.nparams = 3

        self._node.is_statefull = False
        # self._node.output = output
        # parameters
        self.params = relu_params_t()
        self._node.params = <void*>&self.params
        if self._node.params is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, **kwargs):
        super().__init__()

        # Set node parameters
        self.params.gain = kwargs.pop("gain")
        self.params.sign = kwargs.pop("sign")
        self.params.offset = kwargs.pop("offset")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def gain(self):
        """ Gain property """
        return (<relu_params_t*> self._node.params)[0].gain

    @gain.setter
    def gain(self, value):
        """ Set gain """
        (<relu_params_t*> self._node.params)[0].gain = value

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef relu_params_t params = (<relu_params_t*> self._node.params)[0]
        return params
