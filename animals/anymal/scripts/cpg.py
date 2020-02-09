""" Example of ANYMAL oscillator model. """

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
# OSCILLATOR_NAMES = ["OSC_{}".format(num) for num in range(N_OSCILLATORS)]
OSCILLATOR_NAMES = (
    "HAA_LF",
    "HAA_RF",
    "HAA_LH",
    "HAA_RH",
    "HFE_LF",
    "HFE_RF",
    "HFE_LH",
    "HFE_RH",
    "KFE_LF",
    "KFE_RF",
    "KFE_LH",
    "KFE_RH",
)

OSCILLATOR_POSITIONS = {
    "HAA_LF": [-1., 2.],
    "HAA_RF": [1., 2.],
    "HAA_LH": [-1., -2.],
    "HAA_RH": [1., -2.],
    "HFE_LF": [-1., 1.],
    "HFE_RF": [1., 1.],
    "HFE_LH": [-1., -1.],
    "HFE_RH": [1., -1.],
    "KFE_LF": [-2., 1.],
    "KFE_RF": [2., 1.],
    "KFE_LH": [-2., -1.],
    "KFE_RH": [2., -1.],
}

for osc in OSCILLATOR_NAMES:
    network.add_node(osc, model="oscillator",
                     f=1.,
                     R=1.,
                     a=10.,
                     x=OSCILLATOR_POSITIONS[osc][0],
                     y=OSCILLATOR_POSITIONS[osc][1],
                     )

#: Connect the oscillators
for a, b in itertools.product(range(4, 8), range(4, 8)):
    if a != b:
        network.add_edge(
            OSCILLATOR_NAMES[a], OSCILLATOR_NAMES[b],
            weight=np.random.uniform(-1, 1),
            phi=np.pi)
network.add_edge(
    OSCILLATOR_NAMES[4], OSCILLATOR_NAMES[0],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)
network.add_edge(
    OSCILLATOR_NAMES[4], OSCILLATOR_NAMES[8],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)

network.add_edge(
    OSCILLATOR_NAMES[5], OSCILLATOR_NAMES[1],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)
network.add_edge(
    OSCILLATOR_NAMES[5], OSCILLATOR_NAMES[9],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)

network.add_edge(
    OSCILLATOR_NAMES[6], OSCILLATOR_NAMES[2],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)
network.add_edge(
    OSCILLATOR_NAMES[6], OSCILLATOR_NAMES[10],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)

network.add_edge(
    OSCILLATOR_NAMES[7], OSCILLATOR_NAMES[3],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)
network.add_edge(
    OSCILLATOR_NAMES[7], OSCILLATOR_NAMES[11],
    weight=np.random.uniform(-1, 1),
    phi=np.pi)

#: Location to save the network
net_dir = "../config/auto_anymal_oscillator.graphml"
nx.write_graphml(network, net_dir)

# #: Initialize network
dt = 0.001  #: Time step
dur = 25
time_vec = np.arange(0, dur, dt)  #: Time
container = Container(dur/dt)
net = NeuralSystem(
    "../config/auto_anymal_oscillator.graphml",
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
state = np.asarray(container.neural.states.log)
neuron_out = np.asarray(container.neural.outputs.log)

#: Show graph
net.visualize_network(edge_labels=False)


plt.figure()
for j in range(N_OSCILLATORS):
    plt.plot(2*j+(state[:, 2*j+1]*np.cos(neuron_out[:, j])))
plt.grid(True)
plt.figure()
plt.plot(state[:, 1::2])
plt.grid(True)
plt.show()
