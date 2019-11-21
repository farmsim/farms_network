#generate figures for presentation

import farms_pylog as pylog
import networkx as nx
import os
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container

network = nx.DiGraph()

#: initialize network parameters
dt = 0.001  #: Time step
dur = 50
time_vec = np.arange(0, dur, dt)  #: Time


container = Container(dur/dt)

num_oscillators = 2
oscillator_names = []

Varr = [-1,1]
for i in range(num_oscillators):
         
        network.add_node(i, model="matsuoka_neuron",nu=0.2)
        oscillator_names.append(str(i))

for a,b in itertools.product(range(num_oscillators), range(num_oscillators)):
    if a != b:
        network.add_edge(
            oscillator_names[a], oscillator_names[b],
            weight=0)
    
#: Location to save the network
net_dir = 'figures.graphml'
nx.write_graphml(network, net_dir)

# #: Initialize network
net = NeuralSystem('figures.graphml')

container.initialize()

net.setup_integrator()


#: Integrate the network
pylog.info('Begin Integration!')

for t in time_vec:
    net.step(dt=dt)

#: Results
state = container.neural.states.log
neuron_out = container.neural.outputs.log

#: Show graph
#net.visualize_network(edge_labels=False)

plt.figure()
plt.plot(neuron_out[:,0])
plt.xlabel('time (ms)')
plt.ylabel('spike rate (spikes/s)')
plt.grid(True)
plt.show()