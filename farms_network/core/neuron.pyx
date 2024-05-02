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

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strcpy, strdup


cdef void ode_rhs_c(
    double time,
    double[:] states,
    double[:] dstates,
    double[:] inputs,
    double[:] weights,
    double[:] noise,
    double drive,
    Neuron neuron
):
    """ Neuron ODE """
    printf("Base implementation of ODE C function \n")


cdef double output_c(double time, double[:] states, Neuron neuron):
    """ Neuron output """
    printf("Base implementation of neuron output C function \n")
    return 0.0


cdef class PyNeuron:
    """ Python interface to Neuron C-Structure"""

    def __cinit__(self):
        self._neuron = <Neuron*>malloc(sizeof(Neuron))
        if self._neuron is NULL:
            raise MemoryError("Failed to allocate memory for Neuron")
        self._neuron.name = NULL
        self._neuron.model_type = NULL
        self._neuron.ode_rhs_c = ode_rhs_c
        self._neuron.output_c = output_c

    def __dealloc__(self):
        if self._neuron.name is not NULL:
            free(self._neuron.name)
        if self._neuron.model_type is not NULL:
            free(self._neuron.model_type)
        if self._neuron is not NULL:
            free(self._neuron)

    def __init__(self, name, model_type, nstates, nparameters, ninputs):
        self.name = name
        self.model_type = model_type
        self.nstates = nstates
        self.nparameters = nparameters
        self.ninputs = ninputs

    # Property methods for name
    @property
    def name(self):
        if self._neuron.name is NULL:
            return None
        return self._neuron.name.decode('UTF-8')

    @name.setter
    def name(self, value):
        if self._neuron.name is not NULL:
            free(self._neuron.name)
        self._neuron.name = strdup(value.encode('UTF-8'))

    # Property methods for model_type
    @property
    def model_type(self):
        if self._neuron.model_type is NULL:
            return None
        return self._neuron.model_type.decode('UTF-8')

    @model_type.setter
    def model_type(self, value):
        if self._neuron.model_type is not NULL:
            free(self._neuron.model_type)
        self._neuron.model_type = strdup(value.encode('UTF-8'))

    # Property methods for nstates
    @property
    def nstates(self):
        return self._neuron.nstates

    @nstates.setter
    def nstates(self, value):
        self._neuron.nstates = value

    # Property methods for nparameters
    @property
    def nparameters(self):
        return self._neuron.nparameters

    @nparameters.setter
    def nparameters(self, value):
        self._neuron.nparameters = value

    # Property methods for ninputs
    @property
    def ninputs(self):
        return self._neuron.ninputs

    @ninputs.setter
    def ninputs(self, value):
        self._neuron.ninputs = value

    # Methods to wrap the ODE and output functions
    def ode_rhs(self, double time, states, dstates, inputs, weights, noise, drive):
        cdef double[:] c_states = states
        cdef double[:] c_dstates = dstates
        cdef double[:] c_inputs = inputs
        cdef double[:] c_weights = weights
        cdef double[:] c_noise = noise
        # Call the C function directly
        self._neuron.ode_rhs_c(
            time, c_states, c_dstates, c_inputs, c_weights, c_noise, drive, self._neuron[0]
        )

    def output(self, double time, states):
        cdef double[:] c_states = states
        # Call the C function and return its result
        return self._neuron.output_c(
            time, c_states, self._neuron[0]
        )
