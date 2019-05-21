from farms_network_generator.leaky_integrator cimport LeakyIntegrator

ctypedef double real

cdef class NetworkGenerator(object):
    cdef:
        dict __dict__
        LeakyIntegrator[:] c_neurons
    cdef:
        void c_step(self)
        real[:] c_ode(self, real t, real[:] state)
