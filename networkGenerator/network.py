"""This class implements the network of different neurons."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
from scipy.integrate import ode

import biolog
import neuron

G = nx.DiGraph()  #: Create graph
neuron1 = neuron.LeakyIntegrateFire()

# G.add_node(neuron.LeakyIntegrateFire())

G.add_node('neuron1', data=neuron1)
G.add_node('input', data=0.0)
G.add_node('output', data=0.0)

G.add_edge('input', 'neuron1', object=neuron1.membrane, weight=10.)
G.add_edge('neuron1', 'neuron1', object=neuron1.membrane)
G.add_edge('neuron1', 'output')

pos = nx.spring_layout(G)  # positions for all nodes

for node in G.nodes():
    biolog.debug('Node : {}'.format(node))

for edge in G.edges(['neuron1']):
    biolog.debug('Edge : {}'.format(edge))

for node in nx.all_neighbors(G, 'neuron1'):
    biolog.debug(node)

#:  Sparse Matrix
mat = nx.to_scipy_sparse_matrix(G, nodelist=['input', 'neuron1', 'output'])
biolog.debug('Sparse Matrix : {}'.format(mat))

#: Integration
s = ode(G.node['neuron1']['data'].ode)
integrators = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
methods = ['adams', 'bdf']
s.set_integrator(integrators[0],
                 method=methods[0],
                 with_jacobian=False)
t = np.arange(0, 1, 0.01)
t0 = t[0]
dt = t[1] - t0
y0 = 0.0
y = [y0]
_in = 3.0
s.set_initial_value(y0, t0).set_f_params(_in)
while s.successful() and s.t < t[-1]:
    _in = t
    s.integrate(s.t + dt)
    y.append(s.y)
nx.draw(G, with_labels=True)

# plt.figure('neuron')
# plt.plot(t, y)
# plt.plot(t, neuron1.activation(y))
# plt.grid('on')

# plt.figure('networkx')
# plt.draw()
# plt.show()
