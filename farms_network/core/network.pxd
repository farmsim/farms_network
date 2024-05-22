from .neuron cimport Neuron


cdef struct Network:
    unsigned int nstates
    Neuron* neurons
    void step()
    void ode_c()


# cdef void ode_c(
#     Neuron *neurons,
#     unsigned int iteration,
#     unsigned int nneurons,
#     double[:] dstates
# ):
#     """ Network step function """
#     cdef Neuron neuron
#     cdef NeuronData neuron_data
#     cdef unsigned int j
#     cdef nneurons = sizeof(neurons)/sizeof(neuron)

#     # double[:, :] states
#     # double[:, :] dstates
#     # double[:, :] inputs
#     # double[:, :] weights
#     # double[:, :] noise

#     for j in range(nneurons):
#         neuron_data = network_data[j]
#         neurons[j].ode_rhs_c(
#             neuron_data.curr_state,
#             dstates,
#             inputs,
#             weights,
#             noise,
#             drive,
#             neurons[j]
#         )


cdef class PyNetwork:
    """ Python interface to Network ODE """

    cdef:
        Network *_network
        unsigned int nneurons
        list neurons
        Neuron **c_neurons

    cpdef void test(self)
