# distutils: language = c++

from ..numeric.system cimport SDESystem
from libc.math cimport sqrt as csqrt
from libcpp.random cimport mt19937, normal_distribution


cdef struct OrnsteinUhlenbeckParameters:
    double* mu
    double* sigma
    double* tau
    mt19937* random_generator
    normal_distribution[double]* distribution


cdef class OrnsteinUhlenbeck(SDESystem):

    cdef:
        double timestep
        unsigned int n_dim
        OrnsteinUhlenbeckParameters* parameters

    cdef void evaluate_a(self, double time, double timestep, double[:] states, double[:] drift) noexcept
    cdef void evaluate_b(self, double time, double timestep, double[:] states, double[:] diffusion) noexcept
