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
    #: Remove Spine nodes
    network.remove_nodes_from(
        [
            'Cervical_flexion',
            'Cervical_extension',
            # 'Thoracic_flexion',
            # 'Thoracic_extension',
            # 'Lumbar_flexion',
            # 'Lumbar_extension',
        ]
    )
    #: Connect hind-fore limbs
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

    _ed = list(
        network.edges([
            'Lumbar_extension',
            'Lumbar_flexion',
            'Thoracic_extension',
            'Thoracic_flexion'
        ])
    )
    network.remove_edges_from(
        _ed
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
    print(np.asarray(container.neural.states.values))
    x0 = np.random.uniform(
        -1, 1, np.shape(np.asarray(container.neural.states.values))
    )
    net.setup_integrator(list(x0))

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

    plt.figure()
    p_rse = container.neural.states.get_parameter_index(
        'phase_RShoulder_extension')
    p_rsf = container.neural.states.get_parameter_index(
        'phase_RShoulder_flexion')
    p_lse = container.neural.states.get_parameter_index(
        'phase_LShoulder_extension')
    p_lsf = container.neural.states.get_parameter_index(
        'phase_LShoulder_flexion')
    a_rse = container.neural.states.get_parameter_index(
        'amp_RShoulder_extension')
    a_rsf = container.neural.states.get_parameter_index(
        'amp_RShoulder_flexion')
    a_lse = container.neural.states.get_parameter_index(
        'amp_LShoulder_extension')
    a_lsf = container.neural.states.get_parameter_index(
        'amp_LShoulder_flexion')
    plt.plot((state[:, a_rse]*np.sin(state[:, p_rse])))
    plt.plot((2+state[:, a_rsf]*np.sin(state[:, p_rsf])))
    plt.plot((state[:, a_lse]*np.sin(state[:, p_lse])))
    plt.plot((2+state[:, a_lsf]*np.sin(state[:, p_lsf])))
    plt.legend(('RSE', 'RSF', 'LSE', 'LSF'))
    plt.show()

if __name__ == '__main__':
    main()
