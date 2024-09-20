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

from .data_cy import NetworkDataCy, StatesArrayCy


class NetworkData(NetworkDataCy):
    """ Network data """

    def __init__(self, nstates, states):
        """ Network data structure """

        super().__init__(
            nstates,
            states
        )

        _connectivity = None
        _states = None
        _dstates = None
        _outputs = None
        _neurons = None


class NeuronData(NeuronDataCy):
    """ Base class for representing an arbitrary neuron data """

    def __init__(self):
        """Neuron data initialization """

        super().__init__()

        self.states = None
        self.output = None
        self.variables = None
        self.user = None


class StatesArray(StatesArrayCy):
    """ State array data """

    def __init__(self, array):
        super().__init__(array)


def main():

    data = NetworkData(100)


if __name__ == '__main__':
    main()