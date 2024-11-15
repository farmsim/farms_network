from ..numeric.system cimport ODESystem


cdef class OrnsteinUhlenbeck(ODESystem):

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
