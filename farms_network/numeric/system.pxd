cdef class ODESystem:

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept


cdef class SDESystem:

    cdef void evaluate_a(self, double time, double[:] states, double[:] derivatives) noexcept
    cdef void evaluate_b(self, double time, double[:] states, double[:] derivatives) noexcept
