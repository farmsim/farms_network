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
    )
    net_dir = "../config/quadruped_locomotion.graphml"
    network = controller_gen.network
    #: EDIT THE GENERIC CONTROLLER
    #: Remove MTP nodes
    network.remove_nodes_from(
        [
            'LMtp_flexion',
            'RMtp_flexion',
            'LMtp_extension',
            'RMtp_extension'
        ]
    )
    #: Remove Palm nodes
    network.remove_nodes_from(
        [
            'LPalm_flexion',
            'RPalm_flexion',
            'LPalm_extension',
            'RPalm_extension'
        ]
    )
    #: Remove Head nodes
    network.remove_nodes_from(
        [
            'Head_flexion',
            'Head_extension',
        ]
    )
    #: Remove Cervical nodes
    network.remove_nodes_from(
        [
            'Cervical_flexion',
            'Cervical_extension',
        ]
    )
    #: Connect hind-fore limbsw
    weight = 50.0
    AgnosticController.add_mutual_connection(
        network,
        'LHip_flexion',
        'RHip_flexion',
        weight=weight,
        phi=np.pi
    )
    AgnosticController.add_mutual_connection(
        network,
        'LShoulder_flexion',
        'RShoulder_flexion',
        weight=weight,
        phi=np.pi
    )
    AgnosticController.add_mutual_connection(
        network,
        'LHip_flexion',
        'RShoulder_flexion',
        weight=weight,
        phi=0.0
    )
    AgnosticController.add_mutual_connection(
        network,
        'RHip_flexion',
        'LShoulder_flexion',
        weight=weight,
        phi=0.0
    )
    AgnosticController.add_mutual_connection(
        network,
        'RHip_flexion',
        'RShoulder_flexion',
        weight=weight,
        phi=np.pi/2
    )
    AgnosticController.add_mutual_connection(
        network,
        'LHip_flexion',
        'LShoulder_flexion',
        weight=weight,
        phi=np.pi/2
    )
    AgnosticController.add_mutual_connection(
        network,
        'LHip_extension',
        'RHip_extension',
        weight=weight,
        phi=np.pi
    )
    AgnosticController.add_mutual_connection(
        network,
        'LShoulder_extension',
        'RShoulder_extension',
        weight=weight,
        phi=np.pi
    )
    AgnosticController.add_mutual_connection(
        network,
        'LHip_extension',
        'RShoulder_extension',
        weight=weight,
        phi=0.0
    )
    AgnosticController.add_mutual_connection(
        network,
        'RHip_extension',
        'LShoulder_extension',
        weight=weight,
        phi=0.0
    )
    AgnosticController.add_mutual_connection(
        network,
        'RHip_extension',
        'RShoulder_extension',
        weight=weight,
        phi=np.pi/2
    )
    AgnosticController.add_mutual_connection(
        network,
        'LHip_extension',
        'LShoulder_extension',
        weight=weight,
        phi=np.pi/2
    )

    nx.write_graphml(network, net_dir)

    # #: Initialize network
    dt = 0.001  #: Time step
    dur = 2
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
    print(net.graph.number_of_nodes())
    net.visualize_network(
        edge_labels=False,
        node_size=1000        
    )
    nosc = net.network.graph.number_of_nodes()

    plt.figure()
    for j in range(nosc):
        plt.plot(j + (state[:, 2*j+1]*np.sin(neuron_out[:, j])))
    plt.legend(names)
    plt.grid(True)

    plt.figure()
    plt.plot(state[:, 0::2])
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
