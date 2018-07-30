"""This class implements the network of different neurons."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import biolog
import neuron
import scipy as sp
from scipy.integrate import ode
import neuron

class NetworkGenerator(object):
    """Generate Network based on graphml format.
    """
    def __init__(self):
        super(NetworkGenerator, self).__init__()
        self._graph = None
        self._net_ode = None

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

    def generate_network(self):
        """Generate the neural network model."""
        for node, data in self.graph.node.items():
            if data['type'] == 'interneuron':
                self.graph.node[node]['neuron'] = neuron.LIF_Interneuron()
            elif data['type'] == 'motorneuron':
                self.graph.node[node]['neuron'] = neuron.LIF_Motorneuron()
            else:
                raise TypeError('Undefined neuron type {}'.format(
                    data['type']))
            
        return self.graph

    def network_dydt(self, t, states):
        """
        Generate the ODE for whole network.
        """
        #: Get the states of each node
        _new_state = []
        for node, data in self.graph.node.items():
            if data['type'] == 'interneuron':
                _dat = data['neuron'].ode(0.0, [-50, 0], 0.0)
            elif data['type'] == 'motorneuron':
                _dat = data['neuron'].ode(0.0, [ -50, 0, 0.0, 0.0, 0.0], 0.0)
            _new_state.extend(_dat)
        return _new_state
        
def main():
    net_gen = NetworkGenerator()
    G = net_gen.read_graph('./conf/stick_insect_cpg_v1.graphml')
    pos = {}
    for n, data in G.node.items():
        pos[n] = (data['x'], data['y'])
    net_gen.generate_network()

    #: Integration
    s = ode(net_gen.network_dydt)
    integrators = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
    methods = ['adams', 'bdf']
    s.set_integrator(integrators[0],
                     method=methods[0],
                     with_jacobian=False)
    t = np.arange(0, 10, 0.01)
    t0 = t[0]
    dt = t[1] - t0
    y0 = np.random.rand(36)
    y = []
    time = []
    _in = 0.0
    s.set_initial_value(y0, t0).set_f_params(_in)
    while s.successful() and s.t < t[-1]:
        biolog.info('Integrating time {}'.format(s.t))
        _in = t
        s.integrate(s.t + dt)
        y.append(s.y)
    nx.draw(G, pos=pos,
            with_labels=True, node_size=1000)
    plt.draw()
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    main()
