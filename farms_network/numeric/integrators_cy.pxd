from farms_core.array.array_cy cimport DoubleArray1D

from .system cimport ODESystem

include 'types.pxd'


cdef class RK4Solver:
    cdef:
        DoubleArray1D k1
        DoubleArray1D k2
        DoubleArray1D k3
        DoubleArray1D k4
        DoubleArray1D states_tmp

        Py_ssize_t dim
        double dt

    cdef void step(self, ODESystem sys, double time, double[:] state) noexcept
