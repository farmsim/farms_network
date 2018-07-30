"""This class implements the network of different neurons."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import biolog
import neuron
import scipy as sp
from scipy.integrate import ode
import neuron


class NetworkXModel(object):
    """Generate Network based on graphml format.
    """

    def __init__(self):
        super(NetworkXModel, self).__init__()
        self._graph = None  #: NetworkX graph
        self.pos = {}   #: Neuron positions

    @property
    def graph(self):
        """Get the graph object  """
        return self._graph

    @graph.setter
    def graph(self, value):
        """
        Parameters
        ----------
        value : <nx.GraphObject>
            Graph object
        """
        self._graph = value

    def read_graph(self, path):
        """Read graph from the file path.
        Parameters
        ----------
        path : <str>
            File path of the graph

        Returns
        -------
        out : <nx.GraphObject>
            Graph object created by Networkx
        """

        self.graph = nx.read_graphml(path)

        return self.graph

    def read_neuron_position_in_graph(self):
        """ Read the positions of neurons.
        Only if positions are defined. """

        for _neuron, data in self.graph.node.items():
            self.pos[_neuron] = (data.pop('x', None),
                                 data.pop('y', None))
        check_pos_is_none = None in [
            val for x in self.pos.values() for val in x]
        if check_pos_is_none:
            biolog.warning('Missing neuron position information.')
            self.pos = nx.kamada_kawai_layout(self.graph)
        return

    def visualize_network(self):
        """ Visualize the neural network."""
        self.read_neuron_position_in_graph()
        nx.draw(self.graph, pos=self.pos,
                with_labels=True, node_size=1000)
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()


def main():
    net_ = NetworkXModel()

    net_.read_graph(
        './conf/stick_insect_cpg_v1.graphml')

    net_.visualize_network()


if __name__ == '__main__':
    main()
