"""Generate Neural Network."""
import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

import biolog
from networkx_model import NetworkXModel
from neuron import IntegrateAndFire, LIF_Danner, LIF_Danner_Nap


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
        self.ode = []  #: Network ODE-RHS
        self.num_ode = 0  #: Number of ode equations
        self.dae = {}  #: Network DAE setup for integrator
        self.opts = {}  #: Network integration options
        self.integrator = None  #: CASADI Integrator
        #: pylint: disable=invalid-name
        self.x0 = []  #: Initial state of the network

        #: METHODS
        self.read_graph(graph_file_path)
        # self.network_sparse_matrix()
        # self.generate_neuron_models()

        #: Time Integration
        self.dt = 0.001
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
            #: Generate Neuron Models
            biolog.debug('Generating neuron model : {}'.format(name))
            if neuron['type'] == 'if':
                self.neurons[name] = IntegrateAndFire(name, **neuron)
            elif neuron['type'] == 'lif_danner_nap':
                self.neurons[name] = LIF_Danner_Nap(name, **neuron)
            elif neuron['type'] == 'lif_danner':
                self.neurons[name] = LIF_Danner(name)
            else:
                pass

        return

    def generate_network(self):
        """Generate Network Connections"""
        for name, neuron in self.graph.node.iteritems():
            biolog.debug(
                'Establishing neuron {} network connections'.format(name))
            for pred in self.graph.predecessors(name):
                print('{} -> {}'.format(pred, name))
                self.neurons[name].ode_add_input(
                    self.neurons[pred], self.graph[pred][name]['weight'])
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

    def generate_ode(self):
        """ Generate ode rhs for the network."""
        for idx, neuron in enumerate(self.neurons.values()):
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
        #: For variable time step pylint: disable=invalid-name
        self.dae = {'x': self.states,
                    'p': self.params,
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

     #: pylint: disable=invalid-name
    def setup_integrator(self, x0,
                         integration_method='cvodes',
                         opts=None):
        """Setup casadi integrator."""

        #: Generate DAE
        self.generate_dae()
        #: Generate Options for integration
        self.generate_opts(opts)
        #: Initialize states of the integrator
        self.fin['x0'] = x0
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
