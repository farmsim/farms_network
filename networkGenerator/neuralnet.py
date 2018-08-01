"""Generate Neural Network."""
import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

import biolog
from networkx_model import NetworkXModel
from neuron import IntegrateAndFire as IFneuron


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
        self.network_sparse_matrix()
        self.generate_neuron_models()

        return

    def generate_neuron_models(self):
        """Generate sparse ode matrix for the network."""

        for name, neuron in self.graph.node.iteritems():
            biolog.debug('Generating neuron model : {}'.format(name))
            self.neurons[name] = IFneuron(name, neuron['type'],
                                          neuron['is_ext'])
        self.neurons['N1'].bias = -2.75
        self.neurons['N2'].bias = -1.75
        return

    def generate_network_connections(self):
        """Generate the list of connections for the network."""

        net_mat = cas.SX(np.transpose(self.network_sparse_matrix()))
        neuron_out = [
            n_out.neuron_output() for n_out in self.neurons.values()]
        neuron_out = cas.vertcat(*neuron_out)
        return cas.mtimes(net_mat, neuron_out)

    def generate_states(self):
        """Generate ode states for the network."""

        for neuron in self.neurons.values():
            self.states.extend(neuron.ode_states())
        self.num_states = len(self.states)
        biolog.info(15*'#' +
                    ' STATES : {} '.format(self.num_states) +
                    15*'#' +
                    '\n {}'.format(self.states))
        self.states = cas.vertcat(*self.states)
        return

    def generate_params(self):
        """Generate ode parameters for the network."""

        for neuron in self.neurons.values():
            if neuron.is_ext:
                self.params.extend(neuron.ode_params())
        self.num_params = len(self.params)
        biolog.info(15*'#' +
                    ' PARAMS : {} '.format(self.num_params) +
                    15*'#' +
                    '\n {}'.format(self.params))
        self.params = cas.vertcat(*self.params)
        return

    def generate_ode(self):
        """ Generate ode rhs for the network."""

        net_conn = self.generate_network_connections()
        for idx, neuron in enumerate(self.neurons.values()):
            neuron.ode_add_input(net_conn[idx])
            self.ode.extend(neuron.ode_rhs())
        self.num_ode = len(self.ode)
        biolog.info(15*'#' +
                    ' ODE : {} '.format(self.num_ode) +
                    15*'#' +
                    '\n {}'.format(self.ode))
        self.ode = cas.vertcat(*self.ode)
        return

    def generate_dae(self):
        """ Generate dae for Integration."""

        self.generate_states()
        self.generate_params()
        self.generate_ode()
        #: For variable time step pylint: disable=invalid-name
        T = cas.SX.sym('T')
        self.dae = {'x': self.states,
                    'p': cas.vertcat(self.params, T),
                    'ode': T*self.ode}
        return

    def generate_opts(self, opts):
        """ Generate options for integration."""
        if opts is not None:
            self.opts = opts
        else:
            self.opts = {'tf': 1}
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
        self.x0 = x0
        #: Set up the integrator
        self.integrator = cas.integrator('network',
                                         integration_method,
                                         self.dae, self.opts)
        return

    def step(self, time, params=None):
        """Step integrator."""
        _params = []
        if not params and self.num_params > 0:
            raise ValueError('Undefined external parameters')
        elif self.num_params == 0:
            _params.extend([time])
        else:
            _params.extend(params)
            _params.extend([time])
        states = self.integrator(x0=self.x0, p=_params)['xf']
        return np.array(states)


def main():
    """Main."""
    #: Initialize network
    net_ = NeuralNetGen('./conf/simple_cpg.graphml')
    net_.show_network_sparse_matrix()  #: Print network matrix

    #: Initialize integrator properties
    #: pylint: disable=invalid-name
    x0 = [5, 2]  #: Neuron 1 and 2 membrane potentials

    #: Setup the integrator
    net_.setup_integrator(x0)

    #: Initialize network parameters
    #: pylint: disable=invalid-name
    dt = 0.01  #: Time step
    time = np.arange(0, 10, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time), net_.num_states])

    #: Integrate the network
    for idx, t_ in enumerate(time):
        res[idx] = net_.step(t_)[:, 0]

    #: Results
    net_.save_network_to_dot()  #: Save network to dot file
    net_.visualize_network()  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time, res)
    plt.grid()
    plt.figure()
    plt.title('Phase Plot')
    plt.plot(res[:, 0], res[:, 1])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
