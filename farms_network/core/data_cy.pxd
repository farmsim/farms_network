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


from farms_core.array.array_cy cimport DoubleArray1D, DoubleArray2D


cdef class NetworkDataCy:

    cdef:
        public StatesArrayCy states

    cdef:
        public neurons
        public connectivity


cdef class NeuronDataCy:
    """ Neuron data """


cdef class StatesArrayCy(DoubleArray2D):
    """ State array """


cdef class DStatesArrayCy(DoubleArray2D):
    """ DStates array """


cdef class ParametersArrayCy(DoubleArray2D):
    """ Parameters array """


cdef class OutputsArrayCy(DoubleArray2D):
    """ Outputs array """


cdef class InputsArrayCy(DoubleArray2D):
    """ Inputs array """


cdef class DriveArrayCy(DoubleArray2D):
    """ Drive Array """


# # class User2DArrayCy(DoubleArray2D):
# #     ...