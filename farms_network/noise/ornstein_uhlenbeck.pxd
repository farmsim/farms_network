# distutils: language = c++

from libc.math cimport sqrt as csqrt
from libc.stdint cimport uint_fast32_t, uint_fast64_t

from ..numeric.system cimport SDESystem


cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass random_device:
        ctypedef uint_fast32_t result_type
        random_device()
        result_type operator()()

    cdef cppclass mt19937:
        ctypedef uint_fast32_t result_type
        mt19937()
        mt19937(result_type seed)
        result_type operator()()
        result_type min()
        result_type max()
        void discard(size_t z)
        void seed(result_type seed)

    cdef cppclass mt19937_64:
        ctypedef uint_fast64_t result_type

        mt19937_64()
        mt19937_64(result_type seed)
        result_type operator()()
        result_type min()
        result_type max()
        void discard(size_t z)
        void seed(result_type seed)

    cdef cppclass normal_distribution[T]:
        ctypedef T result_type
        normal_distribution()
        normal_distribution(result_type, result_type)
        result_type operator()[Generator](Generator&)
        result_type min()
        result_type max()


cdef struct OrnsteinUhlenbeckParameters:
    double* mu
    double* sigma
    double* tau
    mt19937_64 random_generator
    normal_distribution[double] distribution


cdef class OrnsteinUhlenbeck(SDESystem):

    cdef:
        double timestep
        unsigned int n_dim
        OrnsteinUhlenbeckParameters* parameters
        normal_distribution[double] distribution
        mt19937_64 random_generator

    cdef void evaluate_a(self, double time, double[:] states, double[:] drift) noexcept
    cdef void evaluate_b(self, double time, double[:] states, double[:] diffusion) noexcept
