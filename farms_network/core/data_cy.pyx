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


cdef class NetworkNoiseCy:
    """ Noise data """

    def __init__(
            self,
            states: NDArray[(Any,), np.double],
            drift: NDArray[(Any,), np.double],
            diffusion: NDArray[(Any,), np.double],
            outputs: NDArray[(Any,), np.double],
    ):
        super().__init__()
        self.states = np.array(states, dtype=np.double)
        self.drift = np.array(drift, dtype=np.double)
        self.diffusion = np.array(diffusion, dtype=np.double)
        self.outputs = np.array(outputs, dtype=np.double)


#############
# Node Data #
#############
cdef class NodeDataCy:
    """ Node data """

    def __init__(self):
        """ nodes data initialization """

        super().__init__()
