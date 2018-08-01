"""Generate Neural Network."""
from networkx_model import NetworkXModel
from neuron import IntegrateAndFire as IFneuron
import biolog
import casadi as cas
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetGen(NetworkXModel):
    """Generate Neural Network.
    """

    def __init__(self, graph_file_path):
        """Initialize."""
        super(NeuralNetGen, self).__init__()
        self.read_graph(graph_file_path)

        self.neurons = {}
        self.states = []
        self.params = []
        self.ode = []
        self.inputs = []
        self.dae = {}
        self.integrator = None
        self.generate_neuron_models()
        self.generate_dae()
        self.setup_integrator()
        return

    def generate_neuron_models(self):
        """Generate sparse ode matrix for the network."""

        for name, neuron in self.graph.node.iteritems():
            # biolog.debug('Generating neuron model : {}'.format(name))
            self.neurons[name] = IFneuron(name, neuron['type'],
                                          neuron['is_ext'])
        self.neurons['N2'].bias = -1.75
        return

    def generate_network_connections(self):
        """Generate the list of connections for the network."""
        net_mat = cas.SX(np.transpose(self.network_sparse_matrix()))
        neuron_out = [
            n_out.n_activation() for n_out in self.neurons.values()]
        neuron_out = cas.vertcat(*neuron_out)
        return cas.mtimes(net_mat, neuron_out)

    def generate_states(self):
        """Generate ode states for the network."""
        for neuron in self.neurons.values():
            self.states.extend(neuron.ode_states())

        biolog.info(15*'#' + ' STATES : {} '.format(len(self.states)) + 15*'#' +
                    '\n {}'.format(self.states))
        self.states = cas.vertcat(*self.states)

    def generate_params(self):
        """Generate ode parameters for the network."""
        for neuron in self.neurons.values():
            if neuron.is_ext:
                self.params.extend(neuron.ode_params())

        biolog.info(15*'#' + ' PARAMS : {} '.format(len(self.params)) + 15*'#' +
                    '\n {}'.formatx(self.params))
        self.params = cas.vertcat(*self.params)

    def generate_ode(self):
        """ Generate ode rhs for the network."""
        net_conn = self.generate_network_connections()
        for idx, neuron in enumerate(self.neurons.values()):
            neuron.ode_add_input(net_conn[idx])
            self.ode.extend(neuron.ode_rhs())

        biolog.info(15*'#' + ' ODE : {} '.format(len(self.ode)) + 15*'#' +
                    '\n {}'.format(self.ode))
        self.ode = cas.vertcat(*self.ode)

    def generate_dae(self):
        self.generate_states()
        self.generate_params()
        self.generate_ode()
        """ Generate dae for integation."""

        #: For variable time step
        T = cas.SX.sym('T')

        self.dae = {'x': self.states,
                    'p': cas.vertcat(self.params, T),
                    'ode': T*self.ode}
        return

    def setup_integrator(self):
        """Setup casadi integrator."""
        opts = {"tf": 1.}  # interval length
        self.integrator = cas.integrator('network', 'cvodes', self.dae, opts)

    def step(self, time):
        """Step integrator."""
        states = self.integrator(x0=0.0, p=[time])['xf']
        return np.array(states)


def main():
    net_ = NeuralNetGen('./conf/simple_cpg.graphml')
    net_.show_network_sparse_matrix()
    dt = 0.01
    time = np.arange(0, 10, dt)
    res = np.empty([len(time), 2])
    for idx, t_ in enumerate(time):
        res[idx] = net_.step(t_)[:, 0]
    net_.save_network_to_dot()
    net_.visualize_network()
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
