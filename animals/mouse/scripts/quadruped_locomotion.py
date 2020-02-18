""" Quadruped cpg locomotion controller. """

import farms_pylog as pylog
import networkx as nx
import os
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container
from farms_sdf.sdf import ModelSDF
from farms_network.utils.agnostic_controller import AgnosticController

def main():
    """ Main. """
    controller_gen = AgnosticController(
        ("../../../../farms_blender/animats/"
         "mouse_v1/design/sdf/mouse_locomotion.sdf"),
        connect_closest_neighbors=False,
        connect_base_nodes=False,
    )    
    net_dir = "../config/quadruped_locomotion.graphml"

    #: EDIT THE GENERIC CONTROLLER
    
    nx.write_graphml(controller_gen.network, net_dir)

    # #: Initialize network
    dt = 0.001  #: Time step
    dur = 10
    time_vec = np.arange(0, dur, dt)  #: Time
    container = Container(dur/dt)
    net = NeuralSystem(
        "../config/quadruped_locomotion.graphml",
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
    # container.dump()
    state = np.asarray(container.neural.states.log)
    neuron_out = np.asarray(container.neural.outputs.log)
    names = container.neural.outputs.names
    parameters = container.neural.parameters
    #: Show graph
    print(net.graph.number_of_edges())
    net.visualize_network(edge_labels=False)
    nosc = net.network.graph.number_of_nodes()
    plt.figure()    
    for j in range(nosc):
        plt.plot((state[:, 2*j+1]*np.sin(neuron_out[:, j])))
    plt.legend(names)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
