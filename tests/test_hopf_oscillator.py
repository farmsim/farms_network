""" Test hopf_oscillator neuron model. """

import itertools
import os
import time

import farms_pylog as pylog
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from farms_container import Container
from farms_network.neural_system import NeuralSystem

#: Define a network graph
network = nx.DiGraph()

dt = 1e-3  #: Time step
dur = 10
time_vec = np.arange(0, dur, dt)  #: Time

#: Define container
container = Container(max_iterations=int(dur/dt))

#: Neuron
#: Create an oscillator for each joint
num_oscillators = 4
oscillator_names = [f'n{num}' for num in range(num_oscillators)]
for neuron_name in oscillator_names:
    network.add_node(
        neuron_name,
        model="hopf_oscillator",
        mu=1.0,
        omega=5.0,
        alpha=5.0,
        beta=50.0,
        x0=0.1,
        y0=0.0
    )

#: Connect edges
connection_matrix = np.asarray(
    [
        [0, -1, 1, -1],
        [-1, 0, -1, 1],
        [-1, 1, 0, -1],
        [1, -1, -1, 0]
    ]
).T

for i, j in zip(*np.nonzero(connection_matrix)):
    network.add_edge(
        oscillator_names[i], oscillator_names[j],
        weight=connection_matrix[i, j]*5
    )

#: Location to save the network
net_dir = './test_config/auto_gen_test_hopf_oscillator.graphml'
nx.write_graphml(network, net_dir)

# #: Initialize network
net = NeuralSystem(
    './test_config/auto_gen_test_hopf_oscillator.graphml',
    container)

container.initialize()

net.setup_integrator()

#: initialize network parameters
#: pylint: disable=invalid-name


#: Integrate the network
pylog.info('Begin Integration!')

start_time = time.time()
for t in time_vec:
    net.step(dt=dt)
    container.neural.update_log()
pylog.info(f"--- {(time.time() - start_time)} seconds ---")

#: Results
state = container.neural.states.log
neuron_out = container.neural.outputs.log

#: Show graph
net.visualize_network(node_labels=True)

plt.figure()
plt.plot(time_vec, neuron_out)
plt.grid(True)
plt.figure()
plt.plot(time_vec, state)
plt.grid(True)
plt.show()
