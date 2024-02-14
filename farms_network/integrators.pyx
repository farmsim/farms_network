import numpy as np


cpdef cnp.ndarray c_rk4(double time, cnp.ndarray[double, ndim=1] state, func, double step_size):
    """ Runge-kutta order 4 integrator """
    cdef cnp.ndarray[double, ndim=1] K1 = np.asarray(func(time, state))
    cdef cnp.ndarray[double, ndim=1] K2 = np.asarray(func(time + step_size/2, state + (step_size/2 * K1)))
    cdef cnp.ndarray[double, ndim=1] K3 = np.asarray(func(time + step_size/2, state + (step_size/2 * K2)))
    cdef cnp.ndarray[double, ndim=1] K4 = np.asarray(func(time + step_size, state + (step_size * K3)))
    cdef cnp.ndarray[double, ndim=1] new_state = state + (K1 + 2*K2 + 2*K3 + K4)*(step_size/6)
    return new_state
