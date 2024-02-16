from farms_network.neural_system import NeuralSystem
""" Quadruped cpg locomotion controller. """

import farms_pylog as pylog
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container
from farms_sdf.sdf import ModelSDF
from farms_network.utils.agnostic_controller import AgnosticController


def main():
    """ Main. """
    controller_gen = AgnosticController(
        ("../../../../farms_models_data/mouse_v1/"
         "design/sdf/mouse_locomotion.sdf"),
        connect_mutual=False
    )
    net_dir = "../config/quadruped_locomotion.graphml"
    network = controller_gen.network

    # EDIT THE GENERIC CONTROLLER
    # Remove nodes
    network.remove_nodes_from(
        ['{}_{}'.format(node, action)
            for node in (
                'LMtp', 'RMtp', 'LPalm', 'RPalm', 'Head',
            'Cervical', 'Lumbar', 'Thoracic'
        )
            for action in ('flexion', 'extension')
        ]
    )

    weight = 5000.0

    # Add mutual connection
    for connect in (
            ('LHip', weight, np.pi),
            ('RHip', weight, np.pi),
            ('LShoulder', weight, np.pi),
            ('RShoulder', weight, np.pi),
    ):
        AgnosticController.add_mutual_connection(
            network,
            '{}_flexion'.format(connect[0]),
            '{}_extension'.format(connect[0]),
            weight=connect[1],
            phi=connect[2]
        )

    # Connect hind-fore limbs
    # # Add central to spine
    for j1, j2, phi in [
            ['LHip', 'RHip', np.pi],
            ['LShoulder', 'RShoulder', np.pi],
            ['LHip', 'LShoulder', np.pi],
            ['RHip', 'RShoulder', np.pi],
            ['LHip', 'RShoulder', 0.0],
            ['RHip', 'LShoulder', 0.0],
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

    # Edit node colors
    for node, data in network.nodes.items():
        data['color'] = 'b'

    nx.write_graphml(network, net_dir)

    # # Initialize network
    dt = 0.001  # Time step
    dur = 1
    time_vec = np.arange(0, dur, dt)  # Time
    container = Container(dur/dt)
    net = NeuralSystem(
        "../config/quadruped_locomotion.graphml",
        container)

    # initialize network parameters
    container.initialize()
    print(np.asarray(container.neural.states.values))
    x0 = np.random.uniform(
        -1, 1, np.shape(np.asarray(container.neural.states.values))
    )
    net.setup_integrator()  # list(x0))

    # Integrate the network
    pylog.info('Begin Integration!')

    for t in time_vec:
        net.step(dt=dt)
        container.update_log()

    # Results
    # container.dump()
    state = np.asarray(container.neural.states.log)
    neuron_out = np.asarray(container.neural.outputs.log)
    names = container.neural.outputs.names
    parameters = container.neural.parameters

    # Show graph
    print(net.graph.number_of_edges())
    print(net.graph.number_of_nodes())
    net.visualize_network(
        node_labels=False,
        edge_labels=True,
        node_size=3e3,
        edge_attribute='phi'
    )
    nosc = net.network.graph.number_of_nodes()

    def get_amp_phase(state, states, name):
        amp = state[:, states.get_parameter_index(
            'amp_{}'.format(name)
        )]
        phase = state[:, states.get_parameter_index(
            'phase_{}'.format(name)
        )]
        return (amp, phase)

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
    # spine = (
    #     'Thoracic',
    # )
    seg = (hind, fore)
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
                    (amp, phase) = get_amp_phase(state, states, name)
                    plt.plot(amp*np.sin(phase), marker)
                    plt.grid(True)
                    legend_names.append(name)
            plt.legend(legend_names)

    # GAIT
    plt.figure()
    plt.title('GAIT')
    (amp, phase) = get_amp_phase(state, states, 'LHip_flexion')
    plt.plot(amp*(np.sin(phase)))
    (amp, phase) = get_amp_phase(state, states, 'LHip_extension')
    plt.plot(amp*(np.sin(phase)))
    (amp, phase) = get_amp_phase(state, states, 'RHip_flexion')
    plt.plot(amp*(2+np.sin(phase)))
    (amp, phase) = get_amp_phase(state, states, 'RHip_extension')
    plt.plot(amp*(2+np.sin(phase)))
    (amp, phase) = get_amp_phase(state, states, 'LShoulder_flexion')
    plt.plot(amp*(4+np.sin(phase)))
    (amp, phase) = get_amp_phase(state, states, 'LShoulder_extension')
    plt.plot(amp*(4+np.sin(phase)))
    (amp, phase) = get_amp_phase(state, states, 'RShoulder_flexion')
    plt.plot(amp*(6+np.sin(phase)))
    (amp, phase) = get_amp_phase(state, states, 'RShoulder_extension')
    plt.plot(amp*(6+np.sin(phase)))
    plt.legend(
        ('LHip_flexion', 'LHip_extension',
         'RHip_flexion', 'RHip_extension',
         'LShoulder_flexion', 'LShoulder_extension',
         'RShoulder_flexion', 'RShoulder_extension',
         )
    )
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
