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

from typing import Dict, Iterable, List

cimport numpy as cnp

import numpy as np


##################################
########## Network data ##########
##################################
cdef class NetworkDataCy:
    """ Network data """

    def __init__(self):
        """ network data initialization """

        super().__init__()


cdef class NetworkStatesCy(DoubleArray1D):
    """ State array """

    def __init__(
            self,
            array: NDArray[(Any,), np.double],
            indices: NDArray[(Any,), np.uintc],
    ):
        super().__init__(array)
        self.indices = np.array(indices, dtype=np.uintc)


cdef class NetworkConnectivityCy:
    """ Connectivity array """

    def __init__(
            self,
            sources: NDArray[(Any,), np.uintc],
            weights: NDArray[(Any,), np.double],
            indices: NDArray[(Any,), np.uintc],
    ):
        super().__init__()
        self.sources = np.array(sources, dtype=np.uintc)
        self.weights = np.array(weights, dtype=np.double)
        self.indices = np.array(indices, dtype=np.uintc)


#############
# Node Data #
#############
cdef class NodeDataCy:
    """ Node data """

    def __init__(
            self,
            states: DoubleArray2D,
            derivatives: DoubleArray2D,
            output: DoubleArray1D,
            external_input: DoubleArray1D,
    ):
        """ nodes data initialization """

        super().__init__()
        self.states = states
        self.derivatives = derivatives
        self.output = output
        self.external_input = external_input
