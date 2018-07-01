"""This class implements the network of different neurons."""

import networkx as nx
import matplotlib.pyplot as plt
import neuron
import biolog
from scipy.integrate import ode
import numpy as np

G = nx.Graph()  #: Create graph
neuron1 = neuron.LeakyIntegrateFire()
G.add_node(neuron.LeakyIntegrateFire())
G.add_node('neuron1', data=neuron1)
G.add_node('input', data=0.0)
G.add_node('output', data=0.0)

G.add_edge('input', 'neuron1', object=neuron1.membrane)
G.add_edge('neuron1', 'output')

biolog.debug(type(G.nodes))

#: Integration
s = ode(G.node['neuron1']['data'].ode)
integrators = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
methods = ['adams', 'bdf']
s.set_integrator(integrators[0],
                method=methods[0],
                 with_jacobian=False)
t = np.arange(0, 1, 0.01)
t0 = t[0]
dt = t[1]-t0
y0 = 0.0
y = [y0]
_in = 1.0
s.set_initial_value(y0, t0).set_f_params(_in)
while s.successful() and s.t < t[-1]:
    _in = t
    s.integrate(s.t+dt)
    y.append(s.y)
nx.draw(G, with_labels=True, font_weight='bold')
# plt.figure()
# plt.plot(t, y)
# plt.show()
# plt.figure()
# plt.draw()
# plt.show()
