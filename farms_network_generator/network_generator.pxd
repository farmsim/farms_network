from farms_network_generator.leaky_integrator cimport LeakyIntegrator

ctypedef double real

cdef class NetworkGenerator(object):
    cdef:
        dict __dict__
    cdef:
        void c_step(self)
