import farms_pylog as pylog
import networkx as nx
import os
import math
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container

#: Define a network graph
network = nx.DiGraph()

#: Define container
container = Container()

num_nodes = 0
p = 0.5
beta_mean = 0.001
gamma = 1/12
beta_h = -1-gamma+math.sqrt(2*gamma**2+2*gamma+1)
print(beta_h)
if beta_mean < beta_h:
	g_c = 1 + beta_mean
else:
	g_c = math.sqrt(1-gamma*(gamma+2*beta_mean)+2*math.sqrt(beta_mean*(gamma**2)*math.sqrt(2*gamma+2*beta_mean+2)))
g = 2*g_c

neuron_names = []

num_neurons = 20
for i in range(num_neurons):
	network.add_node(str(i), model="matsuoka_neuron",c=0,T=1/gamma,nu=np.random.normal(beta_mean,0))
	neuron_names.append(str(i))
	num_nodes += 1


for a,b in itertools.product(range(num_nodes), range(num_nodes)):
	randi = np.random.rand()
	if a!=b:
		network.add_edge(
		neuron_names[a], neuron_names[b],
		weight= np.random.normal(0,g**2/num_nodes))

#: Location to save the network
nx.write_graphml(network, 'neuron_test.graphml')

#: initialize network parameters
#: pylint: disable=invalid-name
dt = 0.01  #: Time step
dur = 100
time_vec = np.arange(0, dur, dt)  #: Time

# #: Initialize network
container = Container(dur/dt)
net = NeuralSystem('neuron_test.graphml', container)
container.initialize()

net.setup_integrator()


#: Integrate the network
pylog.info('Begin Integration!')

for t in time_vec:
    net.step(dt=dt)
    container.update_log()

#: Results
state = container.neural.states.log
neuron_out = container.neural.outputs.log
# : Show graph
#net.visualize_network(edge_labels=True)

plt.figure()
plt.plot(neuron_out)
plt.grid(True)
plt.show()
