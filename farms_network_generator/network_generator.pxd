from farms_network_generator.leaky_integrator cimport LeakyIntegrator
from farms_dae_generator.parameters cimport Parameters

ctypedef double real

cdef class NetworkGenerator(object):
    cdef:
        dict __dict__
        LeakyIntegrator[:] c_neurons
        Parameters x
        Parameters c
        Parameters u
        Parameters p
        Parameters y
    cdef:
        void c_step(self, real[:] inputs)
        c_ode(self, real t, real[:] state)
