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

import sys

import numpy as np
from networkx import DiGraph
from scipy.integrate import ode

from farms_network.network_generator import NetworkGenerator

from .networkx_model import NetworkXModel

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore", UserWarning)


class NeuralSystem(NetworkXModel):
    """Neural System.
    """

    def __init__(self, network_graph, container):
        """ Initialize neural system. """
        super(NeuralSystem, self).__init__()
        self.container = container
        #: Add name-space for neural system data
        neural_table = self.container.add_namespace('neural')
        # self.config_path = config_path
        self.integrator = None
        if type(network_graph) is str:
            self.read_graph(network_graph)
        elif type(network_graph) is DiGraph:
            self.graph = network_graph
        #: Create network
        self.network = NetworkGenerator(self.graph, neural_table)

    def setup_integrator(
            self, x0=None, integrator=u'dopri5', atol=1e-12, rtol=1e-6,
            max_step=0.0, method=u'adams', nsteps=1
    ):
        """Setup system."""
        self.integrator = ode(self.network.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol,
            max_step=max_step,
            # nsteps=1
        )

        if x0 is None:
            # initial_values = np.random.rand(
            #     self.container.neural.states.values
            # )
            self.integrator.set_initial_value(
                self.container.neural.states.values, 0.0
            )
        else:
            self.integrator.set_initial_value(x0, 0.0)

    def step(self, dt=1, update=True):
        """Step ode system. """
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t)
        self.integrator.integrate(self.integrator.t+dt)
