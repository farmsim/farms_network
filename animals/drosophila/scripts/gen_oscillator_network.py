""" Script to generate an oscillator network for drosophila model. """

import farms_pylog as pylog
import networkx as nx
import os
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container
pylog.set_level('error')
#: Define a network graph
network = nx.DiGraph()

#: Define the joints in the fly
#: Generate one oscillator per joint
FLY_LEG_SEGMENTS = ['COXA', 'FEMUR', 'TIBIA', 'TARSUS']
FLY_LEG_SIDES = ['L', 'R'] #: Left, Right
FLY_LEG_PLACEMENTS = ['F', 'M', 'H'] #: 'FRONT', 'MID', 'HIND'

#: Create an oscillator for each joint
num_oscillators = 0
oscillator_names = []
for seg in FLY_LEG_SEGMENTS:
    for placement in FLY_LEG_PLACEMENTS:
        for side in FLY_LEG_SIDES:
            leg = side.lower()+placement.lower()+seg.lower()+"_link_osc"
            network.add_node(leg, model="oscillator",
                             f=np.random.uniform(0, 2),
                             R=np.random.uniform(0, 1),
                             a=np.random.uniform(0, 1))
            oscillator_names.append(leg)
            num_oscillators += 1

#: Connect all nodes to all edges
for a,b in itertools.product(range(num_oscillators), range(num_oscillators)):
    if a != b:
        network.add_edge(
            oscillator_names[a], oscillator_names[b],
            weight=np.random.uniform(0, 1),
            phi=np.random.uniform(0, 1))  
    
#: Location to save the network
net_dir = '../config/auto_gen_fly_oscillator_network.graphml'
nx.write_graphml(network, net_dir)

# #: Initialize network
net = NeuralSystem('../config/auto_gen_fly_oscillator_network.graphml')
container = Container.get_instance()
container.initialize()
net.setup_integrator()

#: initialize network parameters
#: pylint: disable=invalid-name
dt = 0.001  #: Time step
dur = 20
time_vec = np.arange(0, dur, dt)  #: Time

#: Integrate the network
pylog.info('Begin Integration!')

for t in time_vec:
    net.step(dt=dt)

#: Results
state = container.neural.states.log
neuron_out = container.neural.outputs.log

#: Show graph
net.visualize_network(edge_labels=False)

plt.figure()
plt.plot(np.sin(neuron_out))
plt.grid(True)
plt.show()
