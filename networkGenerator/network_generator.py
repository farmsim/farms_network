
""" Generate neural network. """

import casadi as cas
import casadi.tools as cast
import numpy as np
from collections import OrderedDict

import biolog
from dae_generator import DaeGenerator
from networkx_model import NetworkXModel
from neuron import (IntegrateAndFire, LIF_Danner, LIF_Danner_Nap,
                    LIF_Daun_Interneuron, LIF_Daun_Motorneuron)


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
        self.dae = DaeGenerator()
        self.opts = {}  #: Integration parameters
        self.fin = {}
        self.integrator = {}

        #: METHODS
        self.read_graph(graph_file_path)
        self.generate_neurons()
        self.generate_network()

        return

    def generate_neurons(self):
        """Generate the complete neural network.
        Instatiate a neuron model for each node in the graph

        Returns
        -------
        out : <bool>
            Return true if successfully created the neurons
        """

        for name, neuron in sorted(self.graph.node.iteritems()):
            #: Add neuron to list
            biolog.debug(
                'Generating neuron model : {} of type {}'.format(
                    name, neuron['type']))
            #: Generate Neuron Models
            if neuron['type'] == 'if':
                self.neurons[name] = IntegrateAndFire(name, self.dae,
                                                      ** neuron)
            elif neuron['type'] == 'lif_danner_nap':
                self.neurons[name] = LIF_Danner_Nap(name, self.dae,
                                                    **neuron)
            elif neuron['type'] == 'lif_danner':
                self.neurons[name] = LIF_Danner(name, self.dae,
                                                **neuron)
            elif neuron['type'] == 'lif_daun_interneuron':
                self.neurons[name] = LIF_Daun_Interneuron(name,
                                                          self.dae,
                                                          **neuron)
            elif neuron['type'] == 'lif_daun_motorneuron':
                self.neurons[name] = LIF_Daun_Motorneuron(name,
                                                          self.dae,
                                                          **neuron)
            else:
                pass
        return

    def generate_network(self):
        """
        Generate the network.
        """
        for name, neuron in self.neurons.iteritems():
            biolog.debug(
                'Establishing neuron {} network connections'.format(
                    name))
            for pred in self.graph.predecessors(name):
                print('{} -> {}'.format(pred, name))
                neuron.add_ode_input(
                    self.neurons[pred], **self.graph[pred][name])

    def generate_opts(self, opts):
        """ Generate options for integration."""
        if opts is not None:
            self.opts = opts
        else:
            self.opts = {'tf': 0.001,
                         'jit': False,
                         "print_stats": False}
        return

    #: pylint: disable=invalid-name
    def setup_integrator(self,
                         integration_method='idas',
                         opts=None):
        """Setup casadi integrator."""

        #: Generate Options for integration
        self.generate_opts(opts)
        #: Initialize states of the integrator
        self.fin['x0'] = self.dae.x.vals()
        self.fin['p'] = self.dae.u.vals() +\
            self.dae.p.vals() + self.dae.c.vals()
        self.fin['z0'] = self.dae.z.vals()
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
        """Step integrator."""
        res = self.integrator.call(self.fin)
        self.fin['x0'] = res['xf']
        return res


def main():
    """ Main function."""
    net = NetworkGenerator('./conf/motorneuron_daun_test.graphml')
    net.visualize_network()


if __name__ == '__main__':
    main()
