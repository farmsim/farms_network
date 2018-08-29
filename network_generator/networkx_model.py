"""This class implements the network of different neurons."""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import write_dot
import matplotlib.colors as mcolors
import biolog

try:
    from mayavi import mlab
except ImportError:
    biolog.warning(
        'Can not import mayavi. 3D Visualization wont be available')


class NetworkXModel(object):
    """Generate Network based on graphml format.
    """

    def __init__(self):
        """ Initialize. """
        super(NetworkXModel, self).__init__()
        self._graph = None  #: NetworkX graph
        self.pos = {}   #: Neuron positions
        self.edge_pos = {}
        self.color_map = []  #: Neuron color map
        self.color_map_arr = []  #: Neuron color map
        self.color_map_edge = []  #: Neuron edge color map
        self.alpha_edge = []  #: Neuron edge alpha
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
            self.pos[_neuron] = (data.get('x', None),
                                 data.get('y', None))
            self.edge_pos[_neuron] = (data.get('x', None),
                                      data.get('y', None))
        check_pos_is_none = None in [
            val for x in self.pos.values() for val in x]
        if check_pos_is_none:
            biolog.warning('Missing neuron position information.')
            # self.pos = nx.kamada_kawai_layout(self.graph)
            self.pos = nx.spring_layout(self.graph)
            self.edge_pos = self.pos
        return

    def read_neuron_colors_in_graph(self):
        """ Read the neuron display colors."""
        for data in self.graph.node.values():
            self.color_map.extend(data.get('color', 'r'))
            self.color_map_arr.append(mcolors.colorConverter.to_rgb(
                self.color_map[-1]))
        return

    def read_edge_colors_in_graph(self):
        """ Read the neuron display colors."""
        for _, _, attr in self.graph.edges(data=True):
            _weight = attr.get('weight')
            #: pylint: disable=no-member
            if np.sign(_weight) == 1:
                self.color_map_edge.extend('g')
            #: pylint: disable=no-member
            elif np.sign(_weight) == -1:
                self.color_map_edge.extend('r')
            else:
                self.color_map_edge.extend('k')
            self.alpha_edge.append(
                max(np.abs(_weight), 0.1))
        return

    def visualize_network(self,
                          node_size=1500,
                          node_labels=True,
                          edge_labels=True,
                          edge_alpha=True, plt_out=None):
        """ Visualize the neural network."""
        self.read_neuron_position_in_graph()
        self.read_neuron_colors_in_graph()
        self.read_edge_colors_in_graph()

        labels = nx.get_edge_attributes(self.graph, 'weight')
        if plt_out is not None:
            fig = plt_out.figure('Network')
            plt_out.autoscale(True)
            ax = plt_out.gca()
        else:
            fig = plt.figure('Network')
            plt.autoscale(True)
            ax = plt.gca()

        #: Draw Nodes
        _ = nx.draw_networkx_nodes(self.graph, pos=self.pos,
                                   with_labels=True,
                                   node_color=self.color_map,
                                   node_size=node_size,
                                   font_size=6.5,
                                   font_weight='bold',
                                   edge_color='k',
                                   alpha=0.8,
                                   ax=ax)
        if node_labels:
            nx.draw_networkx_labels(self.graph, pos=self.pos,
                                    with_labels=True,
                                    font_size=6.5,
                                    font_weight='bold',
                                    alpha=0.8,
                                    ax=ax)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph,
                                         pos=self.pos,
                                         edge_labels=labels,
                                         font_size=10,
                                         clip_on=True,
                                         ax=ax)
        edges = nx.draw_networkx_edges(self.graph,
                                       pos=self.pos,
                                       node_size=node_size,
                                       edge_color=self.color_map_edge,
                                       width=2.,
                                       arrowsize=10,
                                       ax=ax)
        if edge_alpha:
            for edge in range(self.graph.number_of_edges()):
                edges[edge].set_alpha(self.alpha_edge[edge])

        if plt_out is not None:
            plt_out.draw()
            plt_out.subplots_adjust(
                left=0, right=1, top=1, bottom=0)
            plt_out.grid()
            ax.invert_yaxis()
        else:
            fig.draw()
            ax.invert_yaxis()
            fig.subplots_adjust(
                left=0, right=1, top=1, bottom=0)
            fig.grid()
            fig.show()
        return fig

    def visualize_network_3D(self):
        """ Visualize network in 3D using Mayavi."""

        # reorder nodes from 0,len(G)-1
        G = nx.convert_node_labels_to_integers(self.graph)

        # scalar colors
        scalars = np.array(G.nodes())+5

        mlab.figure(1, bgcolor=(0, 0, 0))
        mlab.clf()
        xpos = [pos[0] for pos in self.pos.values()]
        ypos = [pos[1] for pos in self.pos.values()]
        pts = mlab.points3d(xpos, ypos, np.zeros(np.shape(xpos)),
                            scale_factor=1,
                            scale_mode='none',
                            resolution=20)
        pts.mlab_source.dataset.point_data.scalars = self.color_map_arr
        pts.mlab_source.dataset.lines = np.array(G.edges())
        tube = mlab.pipeline.tube(pts, tube_radius=.05)
        mlab.pipeline.surface(tube)

        # mlab.savefig('mayavi2_spring.png')
        mlab.show()  # interactive window

    def save_network_to_dot(self, name='graph'):
        """ Save network file to dot format."""
        write_dot(self.graph, name + '.dot')
        try:
            os.system('dot -Tpng {0}.dot > {0}.png'.format(name))
        except BaseException:
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
