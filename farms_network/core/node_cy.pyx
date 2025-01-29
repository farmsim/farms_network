""" Node """

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from farms_network.core.options import NodeOptions


cdef void ode(
    double time,
    double* states,
    double* derivatives,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    double noise,
    node_t* node,
    EdgeCy** c_edges,
) noexcept:
    """ node_t ODE """
    printf("Base implementation of ODE C function \n")


cdef double output(
    double time,
    double* states,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    node_t* node,
    EdgeCy** c_edges,
) noexcept:
    """ node_t output """
    printf("Base implementation of output C function \n")
    return 0.0


cdef class NodeCy:
    """ Interface to Node C-Structure """

    def __cinit__(self, name: str, model: str):
        self._node = <node_t*>malloc(sizeof(node_t))
        if self._node is NULL:
            raise MemoryError("Failed to allocate memory for node_t")
        self._node.name = strdup(name.encode('UTF-8'))
        self._node.model = strdup(model.encode('UTF-8'))
        self._node.ode = ode
        self._node.output = output
        self._node.parameters = NULL
        self._node.nparameters = 0
        self._node.ninputs = 0

    def __dealloc__(self):
        if self._node is not NULL:
            if self._node.name is not NULL:
                free(self._node.name)
            if self._node.model is not NULL:
                free(self._node.model)
            if self._node.parameters is not NULL:
                free(self._node.parameters)
            free(self._node)

    # Property methods for name
    @property
    def name(self):
        if self._node.name is NULL:
            return None
        return self._node.name.decode('UTF-8')

    # Property methods for model
    @property
    def model(self):
        if self._node.model is NULL:
            return None
        return self._node.model.decode('UTF-8')

    # Property methods for nstates
    @property
    def nstates(self):
        return self._node.nstates

    # Property methods for ninputs
    @property
    def ninputs(self):
        return self._node.ninputs

    # Property methods for nparameters
    @property
    def nparameters(self):
        return self._node.nparameters

    @property
    def parameters(self):
        """Generic accessor for parameters."""
        if not self._node.parameters:
            raise ValueError("node_t parameters are NULL")
        if self._node.nparameters == 0:
            raise ValueError("No parameters available")

        # The derived class should override this method to provide specific behavior
        raise NotImplementedError("Base class does not define parameter handling")

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

        cdef EdgeCy** c_edges = NULL

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
        cdef EdgeCy** c_edges = NULL
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
