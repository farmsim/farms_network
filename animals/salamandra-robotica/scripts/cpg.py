""" Example of double chain oscillator model. """

import farms_pylog as pylog
import networkx as nx
import os
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container
import time
from tqdm import tqdm

pylog.set_level('debug')

#: Create an oscillator chain


def oscillator_chain(n_oscillators, name_prefix, **kwargs):
    """ Create a chain of n-oscillators. """
    #: Define a network graph
    network = nx.DiGraph()
    oscillator_names = [
        "{}_{}".format(name_prefix, n) for n in range(n_oscillators)]
    #: Oscillators
    f = kwargs.get('f', 1)
    R = kwargs.get('R', 1)
    a = kwargs.get('a', 10)
    origin = kwargs.get('origin', [0, 0])
    for j, osc in enumerate(oscillator_names):
        network.add_node(
            osc, model="oscillator", f=f, R=R, a=a, x=origin[0],
            y=origin[1]+j)
    #: Connect
    phase_diff = kwargs.get('axial_phi', 2*np.pi/n_oscillators)
    weight = kwargs.get('axial_w', 10)
    connections = np.vstack(
        (np.arange(n_oscillators),
         np.roll(np.arange(n_oscillators), -1)))[:, :-1]
    for j in np.arange(n_oscillators-1):
        network.add_edge(
            oscillator_names[connections[0, j]],
            oscillator_names[connections[1, j]],
            weight=weight,
            phi=phase_diff)
        network.add_edge(
            oscillator_names[connections[1, j]],
            oscillator_names[connections[0, j]],
            weight=weight,
            phi=-1*phase_diff)
    return network


def oscillator_double_chain(n_oscillators, **kwargs):
    """ Create a double chain of n-oscillators. """
    kwargs['origin'] = [-0.05, 0]
    left_chain = oscillator_chain(n_oscillators, 'left', **kwargs)
    kwargs['origin'] = [0.05, 0]
    right_chain = oscillator_chain(n_oscillators, 'right', **kwargs)
    double = nx.compose_all((left_chain, right_chain))
    #: Connect double chain
    phase_diff = kwargs.get('anti_phi', np.pi)
    weight = kwargs.get('anti_w', 10)
    for n in range(n_oscillators):
        double.add_edge(
            'left_{}'.format(n),
            'right_{}'.format(n),
            weight=weight,
            phi=phase_diff)
        double.add_edge(
            'right_{}'.format(n),
            'left_{}'.format(n),
            weight=weight,
            phi=phase_diff)
    return double


#: Create double chain
n_oscillators = 10
network = oscillator_double_chain(n_oscillators)

#: Location to save the network
net_dir = "../config/auto_salamandra_robotica_oscillator.graphml"
nx.write_graphml(network, net_dir)

# #: Initialize network
dt = 0.001  #: Time step
dur = 10
time_vec = np.arange(0, dur, dt)  #: Time
container = Container(dur/dt)
net = NeuralSystem(
    "../config/auto_salamandra_robotica_oscillator.graphml",
    container)
#: initialize network parameters
container.initialize()
net.setup_integrator()

#: Integrate the network
pylog.info('Begin Integration!')

start_time = time.time()
for t in tqdm(time_vec):
    net.step(dt=dt)
    container.update_log()
pylog.info("--- %s seconds ---" % (time.time() - start_time))

#: Results
# container.dump()
state = np.asarray(container.neural.states.log)
neuron_out = np.asarray(container.neural.outputs.log)
names = container.neural.outputs.names
#: Show graph
net.visualize_network(
    node_size=500,
    edge_labels=False
)

plt.figure()
nosc = n_oscillators
for j in range(nosc):
    plt.plot(2*j+(state[:, 2*j+1]*np.sin(neuron_out[:, j])))
    plt.plot(2*j+(state[:, 2*(j+nosc)+1]*np.sin(neuron_out[:, nosc+j])))
plt.legend(names)
plt.grid(True)
plt.show()
