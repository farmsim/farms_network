"""Generate Neural Network."""
import casadi as cas
import casadi.tools as cast
import matplotlib.pyplot as plt
import numpy as np

import biolog
from networkx_model import NetworkXModel
from neuron import (IntegrateAndFire, LIF_Danner, LIF_Danner_Nap,
                    LIF_Daun_Interneuron, LIF_Daun_Motorneuron)


class NeuralNetGen(NetworkXModel):
    """Generate Neural Network."""

    def __init__(self, graph_file_path):
        """Initialize.

        Parameters
        ----------
        graph_file_path: <str>
            File path to the graphml structure.
        """
        super(NeuralNetGen, self).__init__()

        #: Attributes
        self.neurons = {}  #: Neurons in the network
        self.states = []  #: Network states
        self.num_states = 0  #: Number of state variables
        self.params = []  #: Network external parameters
        self.num_params = 0  #: Number of parameters
        self.alg_var = []
        self.num_alg_var = 0
        self.alg_eqn = []
        self.ode = []  #: Network ODE-RHS
        self.num_ode = 0  #: Number of ode equations
        self.dae = {}  #: Network DAE setup for integrator
        self.opts = {}  #: Network integration options
        self.integrator = None  #: CASADI Integrator
        #: pylint: disable=invalid-name
        self._x0 = []  #: Initial state of the network

        #: METHODS
        self.read_graph(graph_file_path)
        # self.network_sparse_matrix()
        # self.generate_neuron_models()

        #: Time Integration
        self.dt = 1.
        self.fin = {}

        return

    def generate_neurons(self):
        """Generate the complete neural network.
        Instatiate a neuron model for each node in the graph

        Returns
        -------
        out : <bool>
            Return true if successfully created the neurons
        """

        for name, neuron in self.graph.node.iteritems():
            biolog.debug(
                'Generating neuron model : {} of type {}'.format(
                    name, neuron['type']))
            #: Generate Neuron Models
            if neuron['type'] == 'if':
                self.neurons[name] = IntegrateAndFire(name, **neuron)
            elif neuron['type'] == 'lif_danner_nap':
                self.neurons[name] = LIF_Danner_Nap(name, **neuron)
            elif neuron['type'] == 'lif_danner':
                self.neurons[name] = LIF_Danner(name, **neuron)
            elif neuron['type'] == 'lif_daun_interneuron':
                self.neurons[name] = LIF_Daun_Interneuron(
                    name, **neuron)
            elif neuron['type'] == 'lif_daun_motorneuron':
                self.neurons[name] = LIF_Daun_Motorneuron(
                    name, **neuron)
            else:
                pass

        return

    def generate_network(self):
        """Generate Network Connections"""
        for name, _ in self.graph.node.iteritems():
            biolog.debug(
                'Establishing neuron {} network connections'.format(name))
            for pred in self.graph.predecessors(name):
                print('{} -> {}'.format(pred, name))
                self.neurons[name].ode_add_input(
                    self.neurons[pred], self.graph[pred][name].get(
                        'weight', 0.0), **self.graph[pred][name])
        return

    def generate_states(self):
        """Generate ode states for the network."""

        for neuron in self.neurons.values():
            self.states.extend(neuron.ode_states())
        self.num_states = len(self.states)
        biolog.info(15 * '#' +
                    ' STATES : {} '.format(self.num_states) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.states)]))
        self.states = cas.vertcat(*self.states)
        return

    def generate_params(self):
        """Generate ode parameters for the network."""

        for neuron in self.neurons.values():
            self.params.extend(neuron.ode_params())
        self.num_params = len(self.params)
        biolog.info(15 * '#' +
                    ' PARAMS : {} '.format(self.num_params) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, p) for j, p in enumerate(self.params)]))
        self.params = cas.vertcat(*self.params)
        return

    def generate_algebraic_eqn_var(self):
        """ Generate Algebraic equations and states."""

        for neuron in self.neurons.values():
            self.alg_var.extend(neuron.ode_alg_var())
        self.num_alg_var = len(self.alg_var)
        self.alg_var = cas.vertcat(*self.alg_var)

        for neuron in self.neurons.values():
            self.alg_eqn.extend(neuron.ode_alg_eqn())
        self.alg_eqn = cas.vertcat(*self.alg_eqn)
        return

    def generate_ode(self):
        """ Generate ode rhs for the network."""
        for _, neuron in enumerate(self.neurons.values()):
            self.ode.extend(neuron.ode_rhs())
        self.num_ode = len(self.ode)
        biolog.info(15 * '#' +
                    ' ODE : {} '.format(self.num_ode) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, o) for j, o in enumerate(self.ode)]))
        self.ode = cas.vertcat(*self.ode)
        return

    def generate_dae(self):
        """ Generate dae for Integration."""

        self.generate_states()
        self.generate_params()
        self.generate_ode()
        self.generate_algebraic_eqn_var()
        #: For variable time step pylint: disable=invalid-name
        self.dae = {'x': self.states,
                    'p': self.params,
                    'z': self.alg_var,
                    'alg': self.alg_eqn,
                    'ode': self.ode}
        return

    def generate_opts(self, opts):
        """ Generate options for integration."""
        if opts is not None:
            self.opts = opts
        else:
            self.opts = {'tf': 1.,
                         'jit': False,
                         "print_stats": False}
        return

    def set_init_states(self, x0):
        """Set the initial conditions of the neurons.

        Parameters
        ----------
            x0 : <dict>
                Dictionary containing the initial states of the network
        """

        for neuron in self.neurons.keys():
            self._x0.extend([val for val in x0[neuron]])

        return

    def set_params(self, param):
        """Set the parameters of the neurons.

        Parameters
        ----------
            param : <dict>
                Dictionary containing the network parameters
        """

        _param = []

        for neuron in self.neurons.keys():
            _param.extend([val for val in param[neuron]])
        return _param

    #: pylint: disable=invalid-name
    def setup_integrator(self,
                         integration_method='idas',
                         opts=None):
        """Setup casadi integrator."""

        #: Generate DAE
        self.generate_dae()
        #: Generate Options for integration
        self.generate_opts(opts)
        #: Initialize states of the integrator
        self.fin['x0'] = self._x0
        self.fin['p'] = []
        self.fin['z0'] = cas.DM([])
        self.fin['rx0'] = cas.DM([])
        self.fin['rp'] = cas.DM([])
        self.fin['rz0'] = cas.DM([])

        #: Set up the integrator
        self.integrator = cas.integrator('network',
                                         integration_method,
                                         self.dae, self.opts)
        return

    def step(self, params=None):
        """Step integrator."""
        if params is None:
            self.fin['p'] = []
        else:
            self.fin['p'] = params
        res = self.integrator.call(self.fin)
        self.fin['x0'] = res['xf']
        return res


def main():
    """Main."""
    #: Initialize network
    net_ = NeuralNetGen('./conf/simple_danner_cpg.graphml')
    net_.generate_neurons()
    net_.generate_network()

    #: Initialize integrator properties
    #: pylint: disable=invalid-name
    x0 = [-60, -60, -60, 0, -60, 0]

    # #: Setup the integrator
    net_.setup_integrator(x0)

    #: Initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time = np.arange(0, 20000, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time), net_.num_states])

    #: Integrate the network
    for idx, t in enumerate(time):
        res[idx] = net_.step(params=[t*0.05*1e-3])['xf'].full()[:, 0]

    # #: Results
    net_.save_network_to_dot()  #: Save network to dot file
    # net_.visualize_network()  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time, res)
    plt.legend(tuple(net_.states.elements()))
    plt.grid()

    plt.figure()
    plt.title('Phase Plot')
    plt.plot(res[:, 0], res[:, 2])
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
