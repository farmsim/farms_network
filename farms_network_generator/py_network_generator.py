""" Generate neural network. """

import itertools
from collections import OrderedDict

import casadi as cas

import farms_pylog as biolog
from neuron_factory import NeuronFactory

import farms_dae_generator as farms_dae
from networkx_model import NetworkXModel
biolog.set_level('debug')


class NetworkGenerator(NetworkXModel):
    """ Generate Neural Network.
    """

    def __init__(self, graph_file_path):
        """Initialize.

        Parameters
        ----------
        graph_file_path: <str>
            File path to the graphml structure.
        """
        super(NetworkGenerator, self).__init__()

        #: Attributes
        self.neurons = OrderedDict()  #: Neurons in the network
        self.dae = farms_dae.DaeGenerator()
        self.opts = {}  #: Integration parameters
        self.fin = {}
        self.integrator = {}

        #: METHODS
        self.read_graph(graph_file_path)
        self.generate_neurons()
        self.generate_network()

    def generate_neurons(self):
        """Generate the complete neural network.
        Instatiate a neuron model for each node in the graph

        Returns
        -------
        out : <bool>
            Return true if successfully created the neurons
        """

        factory = NeuronFactory()

        for name, neuron in sorted(self.graph.node.items()):
            #: Add neuron to list
            biolog.debug(
                'Generating neuron model : {} of type {}'.format(
                    name, neuron['model']))
            #: Generate Neuron Models
            _neuron = factory.gen_neuron(neuron['model'])
            self.neurons[name] = _neuron(name, self.dae, **neuron)

    def generate_network(self):
        """
        Generate the network.
        """
        for name, neuron in list(self.neurons.items()):
            biolog.debug(
                'Establishing neuron {} network connections'.format(
                    name))
            for pred in self.graph.predecessors(name):
                print(('{} -> {}'.format(pred, name)))
                neuron.add_ode_input(
                    self.neurons[pred], **self.graph[pred][name])

    def generate_opts(self, opts):
        """ Generate options for integration."""
        if opts is not None:
            self.opts = opts
        else:
            self.opts = {'tf': 1.,
                         'jit': False,
                         "print_stats": False}

    #: pylint: disable=invalid-name

    def setup_integrator(self,
                         integration_method='cvodes',
                         opts=None):
        """Setup casadi integrator."""

        #: Generate Options for integration
        self.generate_opts(opts)
        #: Initialize states of the integrator
        self.fin['x0'] = self.dae.x
        self.fin['p'] = self.dae.u +\
            self.dae.p + self.dae.c
        self.fin['z0'] = self.dae.z
        self.fin['rx0'] = cas.DM([])
        self.fin['rp'] = cas.DM([])
        self.fin['rz0'] = cas.DM([])

        #: Set up the integrator
        self.integrator = cas.integrator('network',
                                         integration_method,
                                         self.dae.generate_dae(),
                                         self.opts)

        return self.integrator

    def step(self):
        """Step integrator.
        Parameters
        ----------
        None
        Returns
        ----------
        res: <dict>
        'xf' : States
        'zf' : Algebraic states
        'rxf' : Residue to warm start integrator states
        'rxf' : Residue to warm start integrator states
        """
        self.fin['p'][:] = list(itertools.chain(*self.dae.params))
        res = self.integrator.call(self.fin)
        self.fin['x0'][:] = res['xf'].full()[:, 0]
        self.fin['z0'][:] = res['zf'].full()[:, 0]
        self.fin['rx0'] = res['rxf']
        self.fin['rz0'] = res['rzf']
        return res


def main():
    """ Main function."""
    net = NetworkGenerator(
        '/Users/tatarama/Documents/EPFL-PhD/Projects/BioRobAnimals/network_generator/tests/integrate_fire/conf/integrate_and_fire_test.graphml')
    import matplotlib.pyplot as plt
    net.visualize_network()
    plt.show()


if __name__ == '__main__':
    main()
