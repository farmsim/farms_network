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
from libcpp.cmath cimport sqrt as cppsqrt

from typing import List

import numpy as np

from ..core.options import OrnsteinUhlenbeckOptions


cdef class OrnsteinUhlenbeck(SDESystem):
    """ Ornstein Uhlenheck parameters """

    def __cinit__(self, noise_options: List[OrnsteinUhlenbeckOptions]):
        """ C initialization for manual memory allocation """

        self.n_dim = len(noise_options)

        self.parameters = <OrnsteinUhlenbeckParameters*>malloc(
            sizeof(OrnsteinUhlenbeckParameters)
        )
        if self.parameters is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck Parameters"
            )

        self.parameters.mu = <double*>malloc(self.n_dim*sizeof(double))
        if self.parameters.mu is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck parameter MU"
            )

        self.parameters.sigma = <double*>malloc(self.n_dim*sizeof(double))
        if self.parameters.sigma is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck parameter SIGMA"
            )

        self.parameters.tau = <double*>malloc(self.n_dim*sizeof(double))
        if self.parameters.tau is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck parameter TAU"
            )

        self.parameters.seed = <unsigned int*>malloc(self.n_dim*sizeof(unsigned int))
        if self.parameters.seed is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck parameter SEED"
            )

        self.parameters.random_generator = <mt19937*>malloc(self.n_dim*sizeof(mt19937))
        if self.parameters.random_generator is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck parameter RANDOM GENERATOR"
            )

        self.parameters.distribution = <normal_distribution[double]*>malloc(
            self.n_dim*sizeof(normal_distribution[double])
        )
        if self.parameters.distribution is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck parameter Distribution"
            )

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

    def __init__(self, noise_options: List[OrnsteinUhlenbeckOptions]):
        super().__init__()
        self.initialize_parameters_from_options(noise_options)

    cdef void evaluate_a(self, double time, double[:] states, double[:] drift) noexcept:
        cdef unsigned int j
        cdef OrnsteinUhlenbeckParameters params = (
            <OrnsteinUhlenbeckParameters>self.parameters[0]
        )
        for j in range(self.n_dim):
            drift[j] = (params.mu[j]-states[j])/params.tau[j]

    cdef void evaluate_b(self, double time, double[:] states, double[:] diffusion) noexcept:
        cdef unsigned int j
        cdef OrnsteinUhlenbeckParameters params = (
            <OrnsteinUhlenbeckParameters>self.parameters[0]
        )
        cdef double noise
        for j in range(self.n_dim):
            noise = params.distribution[j](params.random_generator[j])
            diffusion[j] = params.sigma[j]*cppsqrt(2.0/params.tau[j])*noise

    def py_evaluate_a(self, time, states, drift):
        self.evaluate_a(time, states, drift)
        return drift

    def py_evaluate_b(self, time, states, diffusion):
        self.evaluate_b(time, states, diffusion)
        return diffusion

    def initialize_parameters_from_options(self, noise_options):
        """ Initialize the parameters from noise options  """
        for index in range(self.n_dim):
            noise_option = noise_options[index]
            self.parameters.mu[index] = noise_option.mu
            self.parameters.sigma[index] = noise_option.sigma
            self.parameters.tau[index] = noise_option.tau
            self.parameters.random_generator[index] = mt19937(self.parameters.seed[index])
            # The distribution should always be mean=0.0 and std=1.0
            self.parameters.distribution[index] = normal_distribution[double](0.0, 1.0)
