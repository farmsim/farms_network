cimport numpy as cnp


cpdef cnp.ndarray c_rk4(double time, cnp.ndarray[double] state, func, double step_size)
