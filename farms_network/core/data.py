"""

Main data structure for the network

"""

from pprint import pprint
from typing import Dict, Iterable, List

import numpy as np
from farms_core import pylog
from farms_core.array.array import to_array
from farms_core.array.array_cy import (DoubleArray1D, DoubleArray2D,
                                       IntegerArray1D)
from farms_core.array.types import (NDARRAY_V1, NDARRAY_V1_D, NDARRAY_V2_D,
                                    NDARRAY_V3_D)
from farms_core.io.hdf5 import dict_to_hdf5, hdf5_to_dict

from .data_cy import (NetworkConnectivityCy, NetworkDataCy, NetworkLogCy, NetworkNoiseCy,
                      NetworkStatesCy, NetworkLogStatesCy)
from .options import NetworkOptions, NodeOptions, NodeStateOptions


NPDTYPE = np.float64
NPUITYPE = np.uintc


class NetworkData(NetworkDataCy):
    """ Network data """

    def __init__(
            self,
            times,
            states,
            derivatives,
            connectivity,
            outputs,
            tmp_outputs,
            external_inputs,
            noise,
            nodes,
            **kwargs,
    ):
        """ Network data structure """

        super().__init__()

        self.times = times
        self.states = states
        self.derivatives = derivatives
        self.connectivity = connectivity
        self.outputs = outputs
        self.tmp_outputs = tmp_outputs
        self.external_inputs = external_inputs
        self.noise = noise

        self.nodes: List[NodeData] = nodes

        # assert that the data created is c-contiguous
        assert self.states.array.is_c_contig()
        assert self.derivatives.array.is_c_contig()
        assert self.outputs.array.is_c_contig()
        assert self.tmp_outputs.array.is_c_contig()
        assert self.external_inputs.array.is_c_contig()

    @classmethod
    def from_options(cls, network_options: NetworkOptions):
        """ From options """

        buffer_size = network_options.logs.buffer_size
        times = DoubleArray1D(
            array=np.full(
                shape=buffer_size,
                fill_value=0,
                dtype=NPDTYPE,
            )
        )
        states = NetworkStates.from_options(network_options)
        derivatives = NetworkStates.from_options(network_options)
        connectivity = NetworkConnectivity.from_options(network_options)
        noise = NetworkNoise.from_options(network_options)

        outputs = DoubleArray1D(
            array=np.full(
                shape=(len(network_options.nodes),),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )

        tmp_outputs = DoubleArray1D(
            array=np.full(
                shape=(len(network_options.nodes),),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )

        external_inputs = DoubleArray1D(
            array=np.full(
                shape=len(network_options.nodes),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )
        nodes = [
            NodeData(
                node_options.name,
                NodeStates(states, node_index,),
                NodeOutput(outputs, node_index,),
                NodeExternalInput(external_inputs, node_index,),
            )
            for node_index, node_options in enumerate(network_options.nodes)
        ]
        # nodes = np.array(
        #     [
        #         NodeData.from_options(
        #             node_options,
        #             buffer_size=network_options.logs.buffer_size
        #         )
        #         for node_options in network_options.nodes
        #     ],
        #     dtype=NodeDataCy
        # )
        return cls(
            times=times,
            states=states,
            derivatives=derivatives,
            connectivity=connectivity,
            outputs=outputs,
            tmp_outputs=tmp_outputs,
            external_inputs=external_inputs,
            noise=noise,
            nodes=nodes,
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'times': to_array(self.times.array),
            'states': self.states.to_dict(),
            'derivatives': self.derivatives.to_dict(),
            'connectivity': self.connectivity.to_dict(),
            'outputs': to_array(self.outputs.array),
            'tmp_outputs': to_array(self.tmp_outputs.array),
            'external_inputs': to_array(self.external_inputs.array),
            'noise': self.noise.to_dict(),
            'nodes': {node.name: node.to_dict() for node in self.nodes},
        }

    def to_file(self, filename: str, iteration: int = None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to %s', filename)


class NetworkStates(NetworkStatesCy):

    def __init__(self, array, indices):
        super().__init__(array, indices)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        nstates = 0
        indices = [0,]
        for index, node in enumerate(nodes):
            nstates += node._nstates
            indices.append(nstates)
        return cls(
            array=np.array(np.zeros((nstates,)), dtype=NPDTYPE),
            indices=np.array(indices)
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array),
            'indices': to_array(self.indices),
        }


class NetworkConnectivity(NetworkConnectivityCy):

    def __init__(self, node_indices, edge_indices, weights, index_offsets):
        super().__init__(node_indices, edge_indices, weights, index_offsets)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        edges = network_options.edges

        connectivity = np.full(
            shape=(len(edges), 4),
            fill_value=0,
            dtype=NPDTYPE,
        )
        node_names = [node.name for node in nodes]

        for index, edge in enumerate(edges):
            connectivity[index][0] = int(node_names.index(edge.source))
            connectivity[index][1] = int(node_names.index(edge.target))
            connectivity[index][2] = edge.weight
            connectivity[index][3] = index
        connectivity = np.array(sorted(connectivity, key=lambda col: col[1]))

        node_indices = np.full(
            shape=len(edges),
            fill_value=0,
            dtype=NPDTYPE,
        )
        weights = np.full(
            shape=len(edges),
            fill_value=0,
            dtype=NPDTYPE,
        )
        edge_indices = np.full(
            shape=len(edges),
            fill_value=0,
            dtype=NPDTYPE,
        )
        nedges = 0
        index_offsets = []
        if len(edges) > 0:
            index_offsets.append(0)
            for index, node in enumerate(nodes):
                _node_indices = connectivity[connectivity[:, 1] == index][:, 0].tolist()
                _weights = connectivity[connectivity[:, 1] == index][:, 2].tolist()
                _edge_indices = connectivity[connectivity[:, 1] == index][:, 3].tolist()
                nedges += len(_node_indices)
                index_offsets.append(nedges)
                node_indices[index_offsets[index]:index_offsets[index+1]] = _node_indices
                edge_indices[index_offsets[index]:index_offsets[index+1]] = _edge_indices
                weights[index_offsets[index]:index_offsets[index+1]] = _weights
        return cls(
            node_indices=np.array(node_indices, dtype=NPUITYPE),
            edge_indices=np.array(edge_indices, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
            index_offsets=np.array(index_offsets, dtype=NPUITYPE)
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'node_indices': to_array(self.node_indices),
            'edge_indices': to_array(self.edge_indices),
            'weights': to_array(self.weights),
            'index_offsets': to_array(self.index_offsets),
        }


class NetworkNoise(NetworkNoiseCy):
    """ Data for network noise modeling """

    def __init__(self, states, indices,drift, diffusion, outputs):
        super().__init__(states, indices, drift, diffusion, outputs)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        n_noise_states = 0
        n_nodes = len(nodes)

        indices = []
        # for index, node in enumerate(nodes):
        #     if node.noise and node.noise.is_stochastic:
        #         n_noise_states += 1
        #         indices.append(index)

        return cls(
            states=np.full(
                shape=n_noise_states,
                fill_value=0.0,
                dtype=NPDTYPE,
            ),
            indices=np.array(
                indices,
                dtype=NPUITYPE,
            ),
            drift=np.full(
                shape=n_noise_states,
                fill_value=0.0,
                dtype=NPDTYPE,
            ),
            diffusion=np.full(
                shape=n_noise_states,
                fill_value=0.0,
                dtype=NPDTYPE,
            ),
            outputs=np.full(
                shape=n_nodes,
                fill_value=0.0,
                dtype=NPDTYPE,
            )
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'states': to_array(self.states),
            'indices': to_array(self.indices),
            'drift': to_array(self.drift),
            'diffusion': to_array(self.diffusion),
            'outputs': to_array(self.outputs),
        }


class NetworkLogStates(NetworkLogStatesCy):

    def __init__(self, array, indices):
        super().__init__(array, indices)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        nstates = 0
        indices = [0,]
        buffer_size = network_options.logs.buffer_size
        for index, node in enumerate(nodes):
            nstates += node._nstates
            indices.append(nstates)
        return cls(
            array=np.array(np.zeros((buffer_size, nstates)), dtype=NPDTYPE),
            indices=np.array(indices)
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array),
            'indices': to_array(self.indices),
        }



class NetworkLog(NetworkLogCy):
    """ Network Logs """

    def __init__(
            self,
            times,
            states,
            connectivity,
            outputs,
            external_inputs,
            noise,
            nodes,
            **kwargs,
    ):
        """ Network data structure """

        super().__init__()

        self.times = times
        self.states = states
        self.connectivity = connectivity
        self.outputs = outputs
        self.external_inputs = external_inputs
        self.noise = noise

        self.nodes: List[NodeData] = nodes

        # assert that the data created is c-contiguous
        assert self.states.array.is_c_contig()
        assert self.outputs.array.is_c_contig()
        assert self.external_inputs.array.is_c_contig()

    @classmethod
    def from_options(cls, network_options: NetworkOptions):
        """ From options """

        buffer_size = network_options.logs.buffer_size
        times = DoubleArray1D(
            array=np.full(
                shape=buffer_size,
                fill_value=0,
                dtype=NPDTYPE,
            )
        )
        states = NetworkLogStates.from_options(network_options)
        connectivity = NetworkConnectivity.from_options(network_options)
        noise = NetworkNoise.from_options(network_options)

        outputs = DoubleArray2D(
            array=np.full(
                shape=(buffer_size, len(network_options.nodes)),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )

        external_inputs = DoubleArray2D(
            array=np.full(
                shape=(buffer_size, len(network_options.nodes)),
                fill_value=0,
                dtype=NPDTYPE,
            )
        )
        nodes = [
            NodeData(
                node_options.name,
                NodeStates(states, node_index,),
                NodeOutput(outputs, node_index,),
                NodeExternalInput(external_inputs, node_index,),
            )
            for node_index, node_options in enumerate(network_options.nodes)
        ]

        return cls(
            times=times,
            states=states,
            connectivity=connectivity,
            outputs=outputs,
            external_inputs=external_inputs,
            noise=noise,
            nodes=nodes,
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'times': to_array(self.times.array),
            'states': self.states.to_dict(),
            'connectivity': self.connectivity.to_dict(),
            'outputs': to_array(self.outputs.array),
            'external_inputs': to_array(self.external_inputs.array),
            'noise': self.noise.to_dict(),
            'nodes': {node.name: node.to_dict() for node in self.nodes},
        }

    def to_file(self, filename: str, iteration: int = None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to %s', filename)


class NodeStates:
    def __init__(self, network_states, node_index):
        self._network_states = network_states
        self._node_index = node_index

    @property
    def array(self):
        start = self._network_states.indices[self._node_index]
        end = self._network_states.indices[self._node_index + 1]
        if start == end:
            return None
        return self._network_states.array[:, start:end]


class NodeOutput:
    def __init__(self, network_outputs, node_index):
        self._network_outputs = network_outputs
        self._node_index = node_index

    @property
    def array(self):
        return self._network_outputs.array[:, self._node_index]


class NodeExternalInput:
    def __init__(self, network_external_inputs, node_index):
        self._network_external_inputs = network_external_inputs
        self._node_index = node_index

    @property
    def array(self):
        return self._network_external_inputs.array[:, self._node_index]


class NodeData:
    """ Accesssor for Node Data """
    def __init__(
            self,
            name: str,
            states: "NodeStates",
            output: "NodeOutput",
            external_input: "NodeExternalInput",
    ):
        super().__init__()
        self.name = name
        self.states = states
        self.output = output
        self.external_input = external_input


# class NodeData(NodeDataCy):
#     """ Base class for representing an arbitrary node data """

#     def __init__(
#             self,
#             name: str,
#             states: "NodeStatesArray",
#             output: "NodeOutputArray",
#             external_input: "NodeExternalInputArray",
#     ):
#         """ Node data initialization """

#         super().__init__()
#         self.name = name
#         self.states = states
#         self.output = output
#         self.external_input = external_input

#     @classmethod
#     def from_options(cls, options: NodeOptions, buffer_size: int):
#         """ Node data from class """
#         return cls(
#             name=options.name,
#             states=NodeStatesArray.from_options(options, buffer_size),
#             output=NodeOutputArray.from_options(options, buffer_size),
#             external_input=NodeExternalInputArray.from_options(options, buffer_size),
#         )

#     def to_dict(self, iteration: int = None) -> Dict:
#         """ Concert data to dictionary """
#         return {
#             'states': self.states.to_dict(iteration),
#             'output': to_array(self.output.array),
#             'external_input': to_array(self.output.array),
#         }


# class NodeStatesArray(DoubleArray2D):
#     """ State array data """

#     def __init__(self, array: NDARRAY_V2_D, names: List):
#         super().__init__(array)
#         self.names = names

#     @classmethod
#     def from_options(cls, options: NodeOptions, buffer_size: int):
#         """ State options """
#         nstates = options._nstates
#         if nstates > 0:
#             names = options.state.names
#             array = np.full(
#                 shape=[buffer_size, nstates],
#                 fill_value=0,
#                 dtype=NPDTYPE,
#             )
#         else:
#             names = []
#             array = np.full(
#                 shape=[buffer_size, 0],
#                 fill_value=0,
#                 dtype=NPDTYPE,
#             )
#         return cls(array=array, names=names)

#     def to_dict(self, iteration: int = None) -> Dict:
#         """ Concert data to dictionary """
#         return {
#             'names': self.names,
#             'array': to_array(self.array)
#         }


# class NodeOutputArray(DoubleArray1D):
#     """ Output array data """

#     def __init__(self, array: NDARRAY_V1_D):
#         super().__init__(array)

#     @classmethod
#     def from_options(cls, options: NodeOptions, buffer_size: int):
#         """ State options """
#         array = np.full(
#             shape=buffer_size,
#             fill_value=0,
#             dtype=NPDTYPE,
#         )
#         return cls(array=array)


# class NodeExternalInputArray(DoubleArray1D):
#     """ ExternalInput array data """

#     def __init__(self, array: NDARRAY_V1_D):
#         super().__init__(array)

#     @classmethod
#     def from_options(cls, options: NodeOptions, buffer_size: int):
#         """ State options """
#         array = np.full(
#             shape=buffer_size,
#             fill_value=0,
#             dtype=NPDTYPE,
#         )
#         return cls(array=array)


def main():

    data = NetworkData(100)
    print(data.nodes[0].states.names)


if __name__ == '__main__':
    main()
