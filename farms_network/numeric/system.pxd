cdef class ODESystem:

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
