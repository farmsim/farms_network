# distutils: language = c++
# cython: np_pythran=False

import numpy as np

cimport numpy as cnp
from libcpp.cmath cimport sqrt as cppsqrt
from libc.stdio cimport printf


cdef inline double c_noise_current_update(
    double state, OrnsteinUhlenbeckParameters* params
):
    """ Update OrnsteinUhlenbeck process with Euler–Maruyama method (also called the
    Euler method) is a method for the approximate numerical solution of a stochastic
    differential equation (SDE) """

    cdef double noise = params[0].distribution(params[0].random_generator)
    cdef double next_state = (
        state +
        ((params[0].mu-state)*params[0].dt/params[0].tau) +
        params[0].sigma*(cppsqrt((2.0*params[0].dt)/params[0].tau))*noise
    )
    return next_state
