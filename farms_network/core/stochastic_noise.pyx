# distutils: language = c++

"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------
"""


from libc.stdlib cimport free, malloc


cdef class OrnsteinUhlenbeck(SDESystem):
    """ Ornstein Uhlenheck parameters """

    def __cinit__(self, unsigned int n_dim):
        """ C initialization for manual memory allocation """

        self.parameters = <OrnsteinUhlenbeckParameters*>malloc(sizeof(OrnsteinUhlenbeckParameters))
        if self.parameters is NULL:
            raise MemoryError("Failed to allocate memory for OrnsteinUhlenbeck Parameters")

        self.parameters.mu = <double*>malloc(n_dim*sizeof(double))
        if self.parameters.mu is NULL:
            raise MemoryError("Failed to allocate memory for OrnsteinUhlenbeck parameter MU")

        self.parameters.sigma = <double*>malloc(n_dim*sizeof(double))
        if self.parameters.sigma is NULL:
            raise MemoryError("Failed to allocate memory for OrnsteinUhlenbeck parameter SIGMA")

        self.parameters.tau = <double*>malloc(n_dim*sizeof(double))
        if self.parameters.tau is NULL:
            raise MemoryError("Failed to allocate memory for OrnsteinUhlenbeck parameter TAU")

        self.parameters.random_generator = <mt19937*>malloc(n_dim*sizeof(mt19937))
        if self.parameters.random_generator is NULL:
            raise MemoryError("Failed to allocate memory for OrnsteinUhlenbeck parameter RANDOM GENERATOR")

        self.parameters.distribution = <normal_distribution[double]*>malloc(n_dim*sizeof(normal_distribution[double]))
        if self.parameters.distribution is NULL:
            raise MemoryError("Failed to allocate memory for OrnsteinUhlenbeck parameter Distribution")

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """

        if self.parameters.mu is not NULL:
            free(self.parameters.mu)
        if self.parameters.sigma is not NULL:
            free(self.parameters.sigma)
        if self.parameters.tau is not NULL:
            free(self.parameters.tau)
        if self.parameters.random_generator is not NULL:
            free(self.parameters.random_generator)
        if self.parameters.distribution is not NULL:
            free(self.parameters.distribution)
        if self.parameters is not NULL:
            free(self.parameters)

    def __init__(self, options):
        super().__init__()

    cdef void evaluate_a(self, double time, double[:] states, double[:] derivatives) noexcept:
        ...
        # cdef unsigned int j
        # cdef OrnsteinUhlenbeckParameters params = <OrnsteinUhlenbeckParameters>self.OrnsteinUhlenbeckParameters[0]
        # for j range(self.n_dim):
        #     derivatives[j] = (params.mu[j]-states[j])/params.tau[0]

    cdef void evaluate_b(self, double time, double[:] states, double[:] derivatives) noexcept:
        ...
        # cdef unsigned int j
        # cdef OrnsteinUhlenbeckParameters params = (
        #     <OrnsteinUhlenbeckParameters>self.OrnsteinUhlenbeckParameters[0]
        # )
        # for j range(self.n_dim):
        #     derivatives[j] = (params.mu[j]-states[j])/params.tau[0]
        #     params.sigma[j]*(cppsqrt((2.0*params[0].dt)/params.tau[j]))
