cdef class OrnsteinUhlenbeck(ODESystem):
    """ """

    def __init__(self):
        super().__init__()


    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        ...
