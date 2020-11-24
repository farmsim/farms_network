"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------

This class implements the network of different neurons.
"""

import os

import networkx as nx
import numpy as np

import farms_pylog as pylog


class NetworkXModel(object):
    """Generate Network based on graphml format.
    """

    def __init__(self):
        """ Initialize. """
        super(NetworkXModel, self).__init__()
        self.graph = None  #: NetworkX graph
        self.pos = {}   #: Neuron positions
        self.edge_pos = {}
        self.color_map = []  #: Neuron color map
        self.color_map_arr = []  #: Neuron color map
        self.color_map_edge = []  #: Neuron edge color map
        self.alpha_edge = []  #: Neuron edge alpha
        self.edge_style = []  #: Arrow edge style
        self.net_matrix = None

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
        pylog.info('Showing network connectivity matrix')
        pylog.info(self.net_matrix)

    def read_neuron_position_in_graph(self, from_layout=False):
        """ Read the positions of neurons.
        Only if positions are defined. """
        for _neuron, data in list(self.graph.nodes.items()):
            self.pos[_neuron] = (data.get('x', None),
                                 data.get('y', None))
            self.edge_pos[_neuron] = (data.get('x', None),
                                      data.get('y', None))
        check_pos_is_none = None in [
            val for x in list(self.pos.values()) for val in x]
        if check_pos_is_none:
            pylog.warning('Missing neuron position information.')
            # self.pos = nx.kamada_kawai_layout(self.graph)
            # self.pos = nx.spring_layout(self.graph)
            self.pos = nx.shell_layout(self.graph)
            self.edge_pos = self.pos

    def read_neuron_colors_in_graph(self):
        """ Read the neuron display colors."""
        import matplotlib.colors as mcolors
        for data in list(self.graph.nodes.values()):
            self.color_map.extend(data.get('color', 'r'))
            self.color_map_arr.append(mcolors.colorConverter.to_rgb(
                self.color_map[-1]))

    def read_edge_colors_in_graph(self, edge_attribute='weight'):
        """ Read the neuron display colors."""
        max_weight = max(list(dict(self.graph.edges).items()),
                         key=lambda x: abs(x[1][edge_attribute]),
                         default=[{edge_attribute: 0.0}])[-1][edge_attribute]

        max_weight = abs(max_weight)
        for _, _, attr in self.graph.edges(data=True):
            _weight = attr.get(edge_attribute, 0.0)
            #: pylint: disable=no-member
            try:
                _weight_ratio = _weight/max_weight
            except ZeroDivisionError:
                _weight_ratio = 0.0

            if np.sign(_weight_ratio) == 1:
                self.color_map_edge.extend('g')
            #: pylint: disable=no-member
            elif np.sign(_weight_ratio) == -1:
                self.color_map_edge.extend('r')
            else:
                self.color_map_edge.extend('k')
            self.alpha_edge.append(
                max(np.abs(_weight_ratio), 0.1))

    def visualize_network(self,
                          node_size=1500,
                          node_labels=False,
                          edge_labels=False,
                          edge_attribute='weight',
                          edge_alpha=True,
                          plt_out=None,
                          **kwargs
                          ):
        """ Visualize the neural network."""
        self.read_neuron_position_in_graph()
        self.read_neuron_colors_in_graph()
        self.read_edge_colors_in_graph(edge_attribute=edge_attribute)

        if plt_out is not None:
            fig = plt_out.figure('Network')
            plt_out.autoscale(True)
            ax = plt_out.gca()
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure('Network')
            plt.autoscale(True)
            ax = plt.gca()

        #: Draw Nodes
        _ = nx.draw_networkx_nodes(self.graph, pos=self.pos,
                                   node_color=self.color_map,
                                   node_size=node_size,
                                   alpha=kwargs.pop('alpha', 0.75),
                                   ax=ax
                                   )
        if node_labels:
            nx.draw_networkx_labels(
                self.graph,
                pos=self.pos,
                labels={n: name for n, val in nx.get_node_attributes(
                    self.graph, 'name')},
                font_size=kwargs.pop('font_size', 6.5),
                font_weight=kwargs.pop('font_weight', 'bold'),
                font_family=kwargs.pop('font_family', 'sans-serif'),
                alpha=kwargs.pop('alpha', 0.75),
                ax=ax
            )
        if edge_labels:
            labels = {
                ed: round(val, 3)
                for ed, val in nx.get_edge_attributes(
                    self.graph, edge_attribute
                ).items()
            }
            nx.draw_networkx_edge_labels(self.graph,
                                         pos=self.pos,
                                         rotate=False,
                                         edge_labels=labels,
                                         font_size=kwargs.pop(
                                             'font_size', 6.5),
                                         clip_on=True,
                                         ax=ax)
        edges = nx.draw_networkx_edges(self.graph,
                                       pos=self.pos,
                                       node_size=node_size,
                                       edge_color=self.color_map_edge,
                                       width=kwargs.pop('edge_width', 1.),
                                       arrowsize=kwargs.pop('arrow_size', 10),
                                       style=kwargs.pop(
                                           'edge_style', 'dashed'),
                                       arrows=kwargs.pop('arrows', True),
                                       connectionstyle=kwargs.pop(
                                           'connection_style', "arc3,rad=0.3"),
                                       min_source_margin=kwargs.pop(
                                           'min_source_margin', 5),
                                       min_target_margin=kwargs.pop(
                                           'min_target_margin', 5),
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
            plt_out.tight_layout()
        else:
            # fig.draw()
            ax.invert_yaxis()
            fig.subplots_adjust(
                left=0, right=1, top=1, bottom=0)
            ax.grid()
        return fig

    def save_network_to_dot(self, name='graph'):
        """ Save network file to dot format."""
        from networkx.drawing.nx_pydot import write_dot
        write_dot(self.graph, name + '.dot')
        try:
            os.system('dot -Tpng {0}.dot > {0}.png'.format(name))
        except BaseException:
            pylog.error('Command not found')


def main():
    """Main.
    Test NetworkXModel Reading and Visualization."""
    net_ = NetworkXModel()
    net_.read_graph(
        './conf/stick_insect_cpg_v1.graphml')
    net_.visualize_network()


if __name__ == '__main__':
    main()
