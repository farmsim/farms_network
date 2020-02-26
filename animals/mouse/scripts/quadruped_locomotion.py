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
    # AgnosticController.add_mutual_connection(
    #     network,
    #     'LHip_flexion',
    #     'RShoulder_flexion',
    #     weight=weight,
    #     phi=0.0
    # )
    # AgnosticController.add_mutual_connection(
    #     network,
    #     'RHip_flexion',
    #     'LShoulder_flexion',
    #     weight=weight,
    #     phi=0.0
    # )
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
    # AgnosticController.add_mutual_connection(
    #     network,
    #     'LHip_extension',
    #     'RShoulder_extension',
    #     weight=weight,
    #     phi=0.0
    # )
    # AgnosticController.add_mutual_connection(
    #     network,
    #     'RHip_extension',
    #     'LShoulder_extension',
    #     weight=weight,
    #     phi=0.0
    # )
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
        network.in_edges([
            'Lumbar_extension',
            'Lumbar_flexion',
            'Thoracic_extension',
            'Thoracic_flexion'
        ])
    )
    _ed.extend(list(
        network.out_edges([
            'Lumbar_extension',
            'Lumbar_flexion',
            'Thoracic_extension',
            'Thoracic_flexion'
        ])
    ))
    network.remove_edges_from(
        _ed
    )

    #: Add connections to spine
    AgnosticController.add_mutual_connection(
            network,
            '{}_extension'.format('Thoracic'),
            '{}_flexion'.format('Thoracic'),
            weight=weight,
            phi=np.pi
        )
    AgnosticController.add_mutual_connection(
            network,
            '{}_extension'.format('Lumbar'),
            '{}_flexion'.format('Lumbar'),
            weight=weight,
            phi=np.pi
        )
    for j1, j2, phi in [
            ['LHip', 'Lumbar', 0.0],
            ['RHip', 'Lumbar', 0.0],
            ['LShoulder', 'Thoracic', np.pi],
            ['RShoulder', 'Thoracic', np.pi],
            ['Lumbar', 'Thoracic', 0.0]
    ]:
        AgnosticController.add_mutual_connection(
            network,
            '{}_extension'.format(j1),
            '{}_extension'.format(j2),
            weight=weight,
            phi=phi
        )
        AgnosticController.add_mutual_connection(
            network,
            '{}_flexion'.format(j1),
            '{}_flexion'.format(j2),
            weight=weight,
            phi=phi
        )

    for node, data in network.nodes.items():
        if 'Shoulder' in node:
            side = node[0]
            action = node.split('_')[-1]
            data['x'] = network.nodes[side+'Hip_'+action]['x']
            data['y'] = 0.05 + -1*network.nodes[side+'Hip_'+action]['y']
        elif 'Elbow' in node:
            side = node[0]
            action = node.split('_')[-1]
            data['x'] = network.nodes[side+'Knee_'+action]['x']
            data['y'] = 0.05 + -1*network.nodes[side+'Knee_'+action]['y']
        elif 'Wrist' in node:
            side = node[0]
            action = node.split('_')[-1]
            data['x'] = network.nodes[side+'Ankle_'+action]['x']
            data['y'] = 0.05 + -1*network.nodes[side+'Ankle_'+action]['y']

    nx.write_graphml(network, net_dir)

    # #: Initialize network
    dt = 0.001  #: Time step
    dur = 4
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
        node_size=3e3
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

    states = container.neural.states
    outputs = container.neural.outputs
    hind = ('Hip', 'Knee', 'Ankle')
    fore = ('Shoulder', 'Elbow', 'Wrist')
    spine = ('Thoracic', 'Lumbar')
    seg = (hind, fore, spine)
    for elem in seg:
        plt.figure()
        for j, joint in enumerate(elem):
            plt.subplot(len(elem), 1, j + 1)
            plt.title('{}'.format(joint))
            legend_names = []
            for side in ('L', 'R'):                
                for a in ('flexion', 'extension'):
                    if a == 'flexion':
                        marker = '--'
                    if a == 'extension':
                        marker = '-.'
                    if 'Lumbar' != joint and 'Thoracic' != joint:
                        name = '{}{}_{}'.format(side, joint, a)
                    else:
                        name = '{}_{}'.format(joint, a)
                    amp = state[:, states.get_parameter_index(
                        'amp_{}'.format(name)
                    )]
                    phase = neuron_out[:, outputs.get_parameter_index(
                        'nout_{}'.format(name)
                    )]
                    plt.plot(amp*np.sin(phase), marker)
                    plt.grid(True)
                    legend_names.append(name)
            plt.legend(legend_names)
    plt.show()


if __name__ == '__main__':
    main()
