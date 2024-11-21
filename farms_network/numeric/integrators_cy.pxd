# distutils: language = c++

from farms_core.array.array_cy cimport DoubleArray1D
from libc.math cimport sqrt as csqrt
from libcpp.random cimport mt19937, normal_distribution

from .system cimport ODESystem, SDESystem

include 'types.pxd'


cdef class RK4Solver:
    cdef:
        DoubleArray1D k1
        DoubleArray1D k2
        DoubleArray1D k3
        DoubleArray1D k4
        DoubleArray1D states_tmp

        unsigned int dim
        double dt

    cdef void step(self, ODESystem sys, double time, double[:] state) noexcept


cdef class EulerMaruyamaSolver:

    cdef:
        cdef void step(self, SDESystem sys, double time, double[:] state) noexcept
