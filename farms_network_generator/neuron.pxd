""" Header for Neuron Base Class. """

cdef class Neuron(object):
    """Base neuron class.
    """

    cdef:
        str model_type

    cdef:
        void c_ode_rhs(self)
        void c_output(self)
