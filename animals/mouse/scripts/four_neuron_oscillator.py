""" Example of four neuron oscillator model. """

import farms_pylog as pylog
import networkx as nx
import os
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container

pylog.set_level('debug')
#: Define a network graph
network = nx.DiGraph()

#: Create an oscillator for each joint
N_OSCILLATORS = 12
OSCILLATOR_NAMES = ["OSC_{}".format(num) for num in range(N_OSCILLATORS)]

for osc in OSCILLATOR_NAMES:
    network.add_node(osc, model="oscillator",
                     f=1.,
                     R=1.,
                     a=np.random.uniform(0, 1))

#: Connect all nodes to all edges
for a,b in itertools.product(range(N_OSCILLATORS), range(N_OSCILLATORS)):
    if a != b:
        network.add_edge(
            OSCILLATOR_NAMES[a], OSCILLATOR_NAMES[b],
            weight=np.random.uniform(-1, 1),
            phi=np.pi)  
    
#: Location to save the network
net_dir = '../config/auto_gen_4_neuron_quadruped.graphml'
nx.write_graphml(network, net_dir)

# #: Initialize network
dt = 0.001  #: Time step
dur = 20
time_vec = np.arange(0, dur, dt)  #: Time
container = Container(dur/dt)
net = NeuralSystem(
    '../config/auto_gen_4_neuron_quadruped.graphml',
    container)
#: initialize network parameters
container.initialize()
net.setup_integrator()

#: Integrate the network
pylog.info('Begin Integration!')

for t in time_vec:
    net.step(dt=dt)
    container.update_log()

#: Results
container.dump()
state = container.neural.states.log
neuron_out = container.neural.outputs.log

#: Show graph
net.visualize_network(edge_labels=False)

plt.figure()
plt.plot(np.sin(neuron_out))
plt.grid(True)
plt.show()
