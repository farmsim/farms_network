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

from .data_cy import NetworkDataCy, NeuronDataCy, NeuronsDataCy


class NetworkData(NetworkDataCy):
    """ Network data """

    def __init__(self):
        """Network data structure"""

        super().__init__()

        self.neurons = None
        self.connectivity = None
        self.states = None
        self.inputs = None
        self.outputs = None


class NeuronsData(NeuronsDataCy):
    """ Neuronal data """

    def __init__(self):
        """ Neurons data """

        super().__init__()



class NeuronData(NeuronDataCy):
    """ Base class for representing an arbitrary neuron data """

    def __init__(self):
        """Neuron data initialization """

        super().__init__()

        self.consts = None
        self.variables = None
