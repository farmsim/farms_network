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

from typing import List

import numpy as np
from farms_core.array.types import (NDARRAY_V1, NDARRAY_V1_D, NDARRAY_V2_D,
                                    NDARRAY_V3_D)

from data_cy import NetworkDataCy, StatesArrayCy
from .options import NodeOptions, NodeStateOptions


class NetworkData(NetworkDataCy):
    """ Network data """

    def __init__(self, nstates,):
        """ Network data structure """

        super().__init__(
            nstates,
            # states
        )

        # self.states = states
        self.nodes: List[NodeData] = [NodeData(),]

        _connectivity = None
        _states = None
        _dstates = None
        _outputs = None
        _nodes = None


class NodeData:
    """ Base class for representing an arbitrary node data """

    def __init__(self, states_arr: StatesArray, out_arr: OutputArray):
        """Node data initialization """

        super().__init__()

        self.states = states_arr
        self.output = out_arr
        self.variables = None

    @classmethod
    def from_options(cls, options: NodeOptions):
        """ Node data from class """
        nstates = options._nstates
        return cls(

        )


class StatesArray(StatesArrayCy):
    """ State array data """

    def __init__(self, array: NDARRAY_V2_D, names: List):
        super().__init__(array)
        self.names = names

    @classmethod
    def from_options(cls, options: NodeStateOptions):
        """ State options """



class OutputArray:
    """ Output array data """

    def __init__(self, array):
        super().__init__(array)


def main():

    data = NetworkData(100)
    print(data.nodes[0].states.names)


if __name__ == '__main__':
    main()
