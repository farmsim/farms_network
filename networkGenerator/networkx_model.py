"""This class implements the network of different neurons."""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import write_dot

import biolog


class NetworkXModel(object):
    """Generate Network based on graphml format.
    """

    def __init__(self):
        super(NetworkXModel, self).__init__()
        self._graph = None  #: NetworkX graph
        self.pos = {}   #: Neuron positions
        self.color_map = []  #: Neuron color map
        self.color_map_edge = []  #: Neuron edge color map
        self.edge_style = []  #: Arrow edge style
        self.net_matrix = None

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

    def network_sparse_matrix(self):
        """Show network connectivity matrix."""
        self.net_matrix = nx.to_scipy_sparse_matrix(self.graph)
        self.net_matrix = self.net_matrix.todense()
        return self.net_matrix

    def show_network_sparse_matrix(self):
        """Show network connectivity matrix."""
        biolog.info('Showing network connectivity matrix')
        biolog.info(self.net_matrix)
        return

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
            # self.pos = nx.kamada_kawai_layout(self.graph)
            self.pos = nx.spring_layout(self.graph)
        return

    def read_neuron_colors_in_graph(self):
        """ Read the neuron display colors."""
        for data in self.graph.node.values():
            self.color_map.extend(data.pop('color', 'r'))
        return

    def read_edge_colors_in_graph(self):
        """ Read the neuron display colors."""
        for _, _, attr in self.graph.edges(data=True):
            if np.sign(attr['weight']) == 1:
                self.color_map_edge.extend('g')
            elif np.sign(attr['weight']) == -1:
                self.color_map_edge.extend('r')
        return

    def visualize_network(self):
        """ Visualize the neural network."""
        self.read_neuron_position_in_graph()
        self.read_neuron_colors_in_graph()
        self.read_edge_colors_in_graph()

        labels = nx.get_edge_attributes(self.graph, 'weight')

        nx.draw_networkx_edge_labels(self.graph,
                                     pos=self.pos,
                                     edge_labels=labels)

        nx.draw(self.graph, pos=self.pos,
                with_labels=True, node_color=self.color_map,
                node_size=1000,
                edge_color=self.color_map_edge,
                arrowstyle='simple',
                alpha=0.5)
        plt.draw()
        plt.gca().invert_yaxis()
        return

    def save_network_to_dot(self, name='graph'):
        """ Save network file to dot format."""
        write_dot(self.graph, name + '.dot')
        try:
            os.system('dot -Tpng {0}.dot > {0}.png'.format(name))
        except:
            biolog.error('Command not found')

        return


def main():
    """Main.
    Test NetworkXModel Reading and Visualization."""
    net_ = NetworkXModel()
    net_.read_graph(
        './conf/stick_insect_cpg_v1.graphml')
    net_.visualize_network()


if __name__ == '__main__':
    main()
