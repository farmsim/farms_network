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

    cdef void step(self, ODESystem sys, double time, double[:] state) noexcept:
        cdef unsigned int i
        cdef double dt2 = self.dt / 2.0
        cdef double dt6 = self.dt / 6.0
        cdef double[:] k1 = self.k1.array
        cdef double[:] k2 = self.k2.array
        cdef double[:] k3 = self.k3.array
        cdef double[:] k4 = self.k4.array
        cdef double[:] states_tmp = self.states_tmp.array

        # Compute k1
        sys.evaluate(time, state, k1)
        for i in range(self.dim):
            self.states_tmp.array[i] = state[i] + (self.dt * k1[i])/2.0

        # Compute k2
        sys.evaluate(time + dt2, self.states_tmp.array, k2)
        for i in range(self.dim):
            self.states_tmp.array[i] = state[i] + (self.dt * k2[i])/2.0

        # Compute k3
        sys.evaluate(time + dt2, self.states_tmp.array, k3)
        for i in range(self.dim):
            self.states_tmp.array[i] = state[i] + self.dt * k3[i]

        # Compute k4
        sys.evaluate(time + 1.0, states_tmp, k4)

        # Update y: y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        for i in range(self.dim):
            state[i] += dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


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
