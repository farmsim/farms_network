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

cimport numpy as cnp

import numpy as np


##################################
########## Network data ##########
##################################
cdef class NetworkDataCy:
    """ Network data """

    def __init__(
            self,
            nstates: int,
            states
            # neurons = None,
            # connectivity = None,
            # inputs = None,
            # drives = None,
            # outputs = None,
    ):
        """ neurons data initialization """

        super().__init__()
        self.states = states
        # self.neurons = neurons
        # self.connectivity = connectivity
        # self.inputs = inputs
        # self.drives = drives
        # self.outputs = outputs


cdef class StatesArrayCy(DoubleArray2D):
    """ State array """

    def __init__(self, array):
        super().__init__(array)


cdef class DStatesArrayCy(DoubleArray2D):
    """ DStates array """


cdef class OutputsArrayCy(DoubleArray2D):
    """ Outputs array """


cdef class InputsArrayCy(DoubleArray2D):
    """ Inputs array """


# # class User2DArrayCy(DoubleArray2D):
# #     ...
