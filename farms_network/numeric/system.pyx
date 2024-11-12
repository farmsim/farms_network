""" Template for an ODE system """


cdef class ODESystem:
    """ ODE System """

    def __init__(self):
        """ Initialize """
        ...

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ Evaluate that needs to filled out by an ODE system """
        ...
