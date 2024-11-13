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

from .data_cy import NetworkConnectivityCy, NetworkDataCy, NetworkStatesCy, NodeDataCy
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
            external_inputs,
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
        self.external_inputs = external_inputs

        self.nodes: np.ndarray[NodeDataCy] = nodes

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
        outputs = DoubleArray1D(
            array=np.full(
                shape=len(network_options.nodes),
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
        nodes = np.array(
            [
                NodeData.from_options(
                    node_options,
                    buffer_size=network_options.logs.buffer_size
                )
                for node_options in network_options.nodes
            ],
            dtype=NodeDataCy
        )
        return cls(
            times=times,
            states=states,
            derivatives=derivatives,
            connectivity=connectivity,
            outputs=outputs,
            external_inputs=external_inputs,
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
            'external_inputs': to_array(self.external_inputs.array),
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
            array=np.array(np.zeros((nstates,)), dtype=np.double),
            indices=np.array(indices)
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array),
            'indices': to_array(self.indices),
        }


class NetworkConnectivity(NetworkConnectivityCy):

    def __init__(self, sources, weights, indices):
        super().__init__(sources, weights, indices)

    @classmethod
    def from_options(cls, network_options: NetworkOptions):

        nodes = network_options.nodes
        edges = network_options.edges

        connectivity = np.full(
            shape=(len(edges), 3),
            fill_value=0,
            dtype=NPDTYPE,
        )
        node_names = [node.name for node in nodes]

        for index, edge in enumerate(edges):
            connectivity[index][0] = int(node_names.index(edge.source))
            connectivity[index][1] = int(node_names.index(edge.target))
            connectivity[index][2] = edge.weight
        connectivity = np.array(sorted(connectivity, key=lambda col: col[1]))

        sources = np.full(
            shape=len(edges),
            fill_value=0,
            dtype=NPDTYPE,
        )
        weights = np.full(
            shape=len(edges),
            fill_value=0,
            dtype=NPDTYPE,
        )
        nedges = 0
        indices = []
        if len(edges) > 0:
            indices.append(0)
            for index, node in enumerate(nodes):
                node_sources = connectivity[connectivity[:, 1] == index][:, 0].tolist()
                node_weights = connectivity[connectivity[:, 1] == index][:, 2].tolist()
                nedges += len(node_sources)
                indices.append(nedges)
                sources[indices[index]:indices[index+1]] = node_sources
                weights[indices[index]:indices[index+1]] = node_weights
        return cls(
            sources=np.array(sources, dtype=np.uintc),
            weights=np.array(weights, dtype=np.double),
            indices=np.array(indices, dtype=np.uintc)
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'sources': to_array(self.sources),
            'weights': to_array(self.weights),
            'indices': to_array(self.indices),
        }


class NodeData(NodeDataCy):
    """ Base class for representing an arbitrary node data """

    def __init__(
            self,
            name: str,
            states: "NodeStatesArray",
            derivatives: "NodeStatesArray",
            output: "NodeOutputArray",
            external_input: "NodeExternalInputArray",
    ):
        """ Node data initialization """

        super().__init__()
        self.name = name
        self.states = states
        self.derivatives = derivatives
        self.output = output
        self.external_input = external_input

    @classmethod
    def from_options(cls, options: NodeOptions, buffer_size: int):
        """ Node data from class """
        return cls(
            name=options.name,
            states=NodeStatesArray.from_options(options, buffer_size),
            derivatives=NodeStatesArray.from_options(options, buffer_size),
            output=NodeOutputArray.from_options(options, buffer_size),
            external_input=NodeExternalInputArray.from_options(options, buffer_size),
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """ Concert data to dictionary """
        return {
            'states': self.states.to_dict(iteration),
            'derivatives': self.derivatives.to_dict(iteration),
            'output': to_array(self.output.array),
            'external_input': to_array(self.output.array),
        }


class NodeStatesArray(DoubleArray2D):
    """ State array data """

    def __init__(self, array: NDARRAY_V2_D, names: List):
        super().__init__(array)
        self.names = names

    @classmethod
    def from_options(cls, options: NodeOptions, buffer_size: int):
        """ State options """
        nstates = options._nstates
        if nstates > 0:
            names = options.state.names
            array = np.full(
                shape=[buffer_size, nstates],
                fill_value=0,
                dtype=np.double,
            )
        else:
            names = []
            array = np.full(
                shape=[buffer_size, 0],
                fill_value=0,
                dtype=np.double,
            )
        return cls(array=array, names=names)

    def to_dict(self, iteration: int = None) -> Dict:
        """ Concert data to dictionary """
        return {
            'names': self.names,
            'array': to_array(self.array)
        }


class NodeOutputArray(DoubleArray1D):
    """ Output array data """

    def __init__(self, array: NDARRAY_V1_D):
        super().__init__(array)

    @classmethod
    def from_options(cls, options: NodeOptions, buffer_size: int):
        """ State options """
        array = np.full(
            shape=buffer_size,
            fill_value=0,
            dtype=np.double,
        )
        return cls(array=array)


class NodeExternalInputArray(DoubleArray1D):
    """ ExternalInput array data """

    def __init__(self, array: NDARRAY_V1_D):
        super().__init__(array)

    @classmethod
    def from_options(cls, options: NodeOptions, buffer_size: int):
        """ State options """
        array = np.full(
            shape=buffer_size,
            fill_value=0,
            dtype=np.double,
        )
        return cls(array=array)


def main():

    data = NetworkData(100)
    print(data.nodes[0].states.names)


if __name__ == '__main__':
    main()
