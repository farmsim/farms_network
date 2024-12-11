import numpy as np

from ..core.options import IntegrationOptions

from libc.stdio cimport printf

NPDTYPE = np.float64


cdef class RK4Solver:

    def __init__ (self, unsigned int dim, double dt):

        super().__init__()
        self.dim = dim
        self.dt = dt
        self.k1 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.k2 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.k3 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.k4 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.states_tmp = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )

    cdef void step(self, ODESystem sys, double time, double[:] states) noexcept:
        cdef unsigned int i
        cdef double dt2 = self.dt / 2.0
        cdef double dt6 = self.dt / 6.0
        cdef double[:] k1 = self.k1.array
        cdef double[:] k2 = self.k2.array
        cdef double[:] k3 = self.k3.array
        cdef double[:] k4 = self.k4.array
        cdef double[:] states_tmp = self.states_tmp.array

        # Compute k1
        sys.evaluate(time, states, k1)

        # Compute k2
        for i in range(self.dim):
            states_tmp[i] = states[i] + (dt2 * k1[i])
        sys.evaluate(time + dt2, states_tmp, k2)

        # Compute k3
        for i in range(self.dim):
            states_tmp[i] = states[i] + (dt2 * k2[i])
        sys.evaluate(time + dt2, states_tmp, k3)

        # Compute k4
        for i in range(self.dim):
            states_tmp[i] = states[i] + self.dt * k3[i]
        sys.evaluate(time + self.dt, states_tmp, k4)

        # Update y: y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        for i in range(self.dim):
            states[i] = states[i] + dt6 * (
                k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
            )


cdef class EulerMaruyamaSolver:

    def __init__ (self, unsigned int dim, double dt):

        super().__init__()
        self.dim = dim
        self.dt = dt
        self.drift = DoubleArray1D(
            array=np.full(shape=self.dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.diffusion = DoubleArray1D(
            array=np.full(shape=self.dim, fill_value=0.0, dtype=NPDTYPE,)
        )

    cdef void step(self, SDESystem sys, double time, double[:] state) noexcept:
        """ Update stochastic noise process with Euler–Maruyama method (also called the
        Euler method) is a method for the approximate numerical solution of a stochastic
        differential equation (SDE) """

        cdef unsigned int i
        cdef double[:] drift = self.drift.array
        cdef double[:] diffusion = self.diffusion.array

        sys.evaluate_a(time, state, drift)
        sys.evaluate_b(time, state, diffusion)
        for i in range(self.dim):
            state[i] += drift[i]*self.dt + csqrt(self.dt)*diffusion[i]
