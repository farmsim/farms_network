""" Template for an ODE system """


cdef class ODESystem:
    """ ODE System """

    def __init__(self):
        """ Initialize """
        ...

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ Evaluate that needs to filled out by an ODE system """
        ...


cdef class SDESystem:
    """ SDE system of the form: dXt = a(Xt,t) dt + b(Xt,t) dW,"""

    def __init__(self):
        """ Initialize """
        ...

    cdef void evaluate_a(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ a(Xt,t) """
        ...

    cdef void evaluate_b(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ b(Xt,t) """
        ...
