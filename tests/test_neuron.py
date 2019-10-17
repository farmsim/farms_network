import farms_pylog as pylog
import networkx as nx
import os
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container

#: Define a network graph
network = nx.DiGraph()

#: Define container
container = Container()

num_oscillators = 0
oscillator_names = []

num_neurons = 6
for i in range(num_neurons):
	network.add_node(str(i), model="matsuoka_neuron")
	oscillator_names.append(str(i))
	num_oscillators += 1


for a,b in itertools.product(range(num_oscillators), range(num_oscillators)):
	if a != b:
		network.add_edge(
		oscillator_names[a], oscillator_names[b],
		weight= 2.5)

#: Location to save the network
nx.write_graphml(network, 'neuron_test.graphml')

# #: Initialize network
net = NeuralSystem('neuron_test.graphml')

container.initialize()

net.setup_integrator()

#: initialize network parameters
#: pylint: disable=invalid-name
dt = 0.01  #: Time step
dur = 50
time_vec = np.arange(0, dur, dt)  #: Time

#: Integrate the network
pylog.info('Begin Integration!')

for t in time_vec:
    net.step(dt=dt)

#: Results
state = container.neural.states.log
neuron_out = container.neural.outputs.log

# : Show graph
net.visualize_network(edge_labels=False)

plt.figure()
plt.plot(neuron_out)
plt.grid(True)
plt.show()
