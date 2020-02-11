""" Test morphed_oscillator neuron model. """

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

dt = 0.001  #: Time step
dur = 10
time_vec = np.arange(0, dur, dt)  #: Time


#: Define container
container = Container(max_iterations=int(dur/dt))

#: Neuron
#: Create an oscillator for each joint
num_oscillators = 0
oscillator_names = ['n1',]
for neuron_name in oscillator_names:
    network.add_node(neuron_name,
                     model="morphed_oscillator",
                     f=0.1)
    num_oscillators += 1

#: Connect all nodes to all edges
for a,b in itertools.product(range(num_oscillators), range(num_oscillators)):
    if a != b:
        network.add_edge(
            oscillator_names[a], oscillator_names[b],
            weight=np.random.uniform(0, 1))
    
#: Location to save the network
net_dir = './test_config/auto_gen_test_morphed_oscillator.graphml'
nx.write_graphml(network, net_dir)

# #: Initialize network
net = NeuralSystem(
    './test_config/auto_gen_test_morphed_oscillator.graphml',
    container)

container.initialize()

net.setup_integrator()

#: initialize network parameters
#: pylint: disable=invalid-name


#: Integrate the network
pylog.info('Begin Integration!')

f_theta_n1 = container.neural.parameters.get_parameter("f_theta_n1")
fd_theta_n1 = container.neural.parameters.get_parameter("fd_theta_n1")
theta_n1 = container.neural.states.get_parameter("theta_n1")

for t in time_vec:
    theta = theta_n1.value
    #: EXAMPLE 1
    # f_theta_n1.value = np.cos(
    #     4*theta+0.4545*np.pi) + np.tanh(
    #         np.cos(10*theta)) + np.tanh(np.sin(3*theta)) + 3.7
    # fd_theta_n1.value = -4*np.sin(4*theta) - 10*(
    #     (np.cosh(np.cos(10*theta)))**-2)*np.sin(10*theta) + 3*(
    #         (np.cosh(np.sin(3*theta)))**-2)*np.cos(3*theta)
    #: EXAMPLE 2
    f_theta_n1.value = np.sin(theta) + 2*np.cos(2*theta - 2.) + 3.6
    fd_theta_n1.value = np.cos(theta) - 4*np.sin(2*theta - 2.)
    net.step(dt=dt)
    container.neural.update_log()

#: Results
state = container.neural.states.log
neuron_out = container.neural.outputs.log

#: Show graph
net.visualize_network(edge_labels=False)

plt.figure()
plt.plot(state[:, 1]*np.cos(state[:, 0]),
         state[:, 1]*np.sin(state[:, 0]))
plt.grid(True)
plt.figure()
plt.plot(state)
plt.grid(True)
plt.show()
