"""Generate Neural Network."""
from networkx_model import NetworkXModel
from neuron import IntegrateAndFire as IFneuron
import biolog
import casadi as cas
import numpy as np


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
        self.generate_states()
        self.generate_params()
        self.generate_ode()

        return

    def generate_neuron_models(self):
        """Generate sparse ode matrix for the network."""

        for name, neuron in self.graph.node.iteritems():
            # biolog.debug('Generating neuron model : {}'.format(name))
            self.neurons[name] = IFneuron(name, neuron['type'],
                                          neuron['is_ext'])
        return

    def generate_network_connections(self):
        """Generate the list of connections for the network."""
        net_mat = cas.SX(np.transpose(self.network_sparse_matrix()))
        neuron_out = [
            n_out.n_activation() for n_out in self.neurons.values()]
        from IPython import embed
        embed()

        return cas.mtimes(net_mat, *neuron_out)

    def generate_states(self):
        """Generate ode states for the network."""
        for neuron in self.neurons.values():
            self.states.extend(neuron.ode_states())

    def generate_params(self):
        """Generate ode parameters for the network."""
        for neuron in self.neurons.values():
            if neuron.is_ext:
                self.params.extend(neuron.ode_params())

    def generate_ode(self):
        """ Generate ode rhs for the network."""
        net_conn = self.generate_network_connections()
        from IPython import embed
        embed()

        for idx, neuron in enumerate(self.neurons.values()):
            neuron.ode_add_input(net_conn[0][idx])
            self.ode.extend(neuron.ode_states())
        print(self.ode[0])

    def setup_integrator(self):
        """Setup casadi integrator."""
        self.integrator = cas.integrator('network', 'idas', self.dae)


def main():
    net_ = NeuralNetGen('./conf/integrate_and_fire_test.graphml')
    net_.visualize_network()


if __name__ == '__main__':
    main()
