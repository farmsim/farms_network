""" Core Data """

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


cdef class NetworkLogCy:
    """ Network Log """

    def __init__(self):
        """ Network Logs initialization """

        super().__init__()


cdef class NetworkStatesCy(DoubleArray1D):
    """ State array """

    def __init__(
            self,
            array: NDArray[(Any,), np.double],
            indices: NDArray[(Any,), np.uintc],
    ):
        super().__init__(array)
        assert self.array.is_c_contig()
        self.indices = np.array(indices, dtype=np.uintc)
        assert self.indices.is_c_contig()


cdef class NetworkLogStatesCy(DoubleArray2D):
    """ State array """

    def __init__(
            self,
            array: NDArray[(Any, Any), np.double],
            indices: NDArray[(Any,), np.uintc],
    ):
        super().__init__(array)
        assert self.array.is_c_contig()
        self.indices = np.array(indices, dtype=np.uintc)
        assert self.indices.is_c_contig()


cdef class NetworkConnectivityCy:
    """ Connectivity array """

    def __init__(
            self,
            node_indices: NDArray[(Any,), np.uintc],
            edge_indices: NDArray[(Any,), np.uintc],
            weights: NDArray[(Any,), np.double],
            index_offsets: NDArray[(Any,), np.uintc],
    ):
        super().__init__()
        self.node_indices = np.array(node_indices, dtype=np.uintc)
        assert self.node_indices.is_c_contig()
        self.edge_indices = np.array(edge_indices, dtype=np.uintc)
        assert self.edge_indices.is_c_contig()
        self.weights = np.array(weights, dtype=np.double)
        assert self.weights.is_c_contig()
        self.index_offsets = np.array(index_offsets, dtype=np.uintc)
        assert self.index_offsets.is_c_contig()


cdef class NetworkNoiseCy:
    """ Noise data """

    def __init__(
            self,
            states: NDArray[(Any,), np.double],
            indices: NDArray[(Any,), np.uintc],
            drift: NDArray[(Any,), np.double],
            diffusion: NDArray[(Any,), np.double],
            outputs: NDArray[(Any,), np.double],
    ):
        super().__init__()
        self.states = np.array(states, dtype=np.double)
        self.indices = np.array(indices, dtype=np.uintc)
        self.drift = np.array(drift, dtype=np.double)
        self.diffusion = np.array(diffusion, dtype=np.double)
        self.outputs = np.array(outputs, dtype=np.double)
