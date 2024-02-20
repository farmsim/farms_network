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
from farms_network.integrators import c_rk4

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
        # Add name-space for neural system data
        neural_table = self.container.add_namespace('neural')
        # self.config_path = config_path
        self.integrator = None
        if isinstance(network_graph, str):
            self.read_graph(network_graph)
        elif isinstance(network_graph, DiGraph):
            self.graph = network_graph
        # Create network
        self.network = NetworkGenerator(self.graph, neural_table)
        self.time = None
        self.state = None

    def setup_integrator(
            self, x0=None, integrator=u'dopri5', atol=1e-12, rtol=1e-6,
            max_step=0.0, method=u'adams'
    ):
        """Setup system."""
        self.integrator = ode(self.network.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol,
            max_step=max_step,
            # nsteps=nsteps
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
        self.state = self.integrator.y
        self.time = 0.0

    def euler(self, time, state, func, step_size=1e-3):
        """ Euler integrator """
        new_state = state + step_size*np.array(func(time, state))
        return new_state

    def rk4(self, time, state, func, step_size=1e-3, n_substeps=1):
        """ Runge-kutta order 4 integrator """
        step_size = step_size/float(n_substeps)
        for j in range(n_substeps):
            K1 = np.array(func(time, state))
            K2 = np.array(func(time + step_size/2, state + (step_size/2 * K1)))
            K3 = np.array(func(time + step_size/2, state + (step_size/2 * K2)))
            K4 = np.array(func(time + step_size, state + (step_size * K3)))
            state = state + (K1 + 2*K2 + 2*K3 + K4)*(step_size/6)
            time += step_size
        return state

    def rk5(self, time, state, func, step_size=1e-3):
        """ Runge-kutta order 5 integrator """
        K1 = np.array(func(time, state))
        K2 = np.array(func(time + step_size/4.0, state + (step_size/4.0 * K1)))
        K3 = np.array(func(time + step_size/4.0, state + (step_size/8.0)*(K1 + K2)))
        K4 = np.array(func(time + step_size/2.0, state - (step_size/2.0 * K2) + (step_size * K3)))
        K5 = np.array(func(time + 3*step_size/4.0, state + (step_size/16.0)*(3*K1 + 9*K4)))
        K6 = np.array(func(time + step_size, state + (step_size/7.0)*(-3*K1 + 2*K2 + 12*K3 + -12*K4 + 8*K5)))
        new_state = np.array(state) + (7/90*K1 + 32/90*K3 + 12/90*K4 + 32/90*K5 + 7/90*K6)*(step_size)
        return new_state

    def step(self, dt=1, update=True):
        """Step ode system. """
        self.time += dt
        self.state = self.rk4(
            self.time, self.state, self.network.ode, step_size=dt, n_substeps=2
        )
        # self.state = c_rk4(
        #     self.time, self.state, self.network.ode, step_size=dt
        # )
        # self.integrator.set_initial_value(self.integrator.y,
        #                                   self.integrator.t)
        # self.integrator.integrate(self.integrator.t+dt)
