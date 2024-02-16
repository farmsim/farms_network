""" cpg locomotion controller. """

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
import yaml

def main():
    """ Main. """
    controller_gen = AgnosticController(
        ("../../../../farms_blender/animats/"
         "drosophila_v1/design/sdf/drosophila_multiDOF.sdf"),
        connect_mutual=False,
        connect_closest_neighbors=False,
        connect_base_nodes=False
    )
    net_dir = "../config/locomotion.graphml"
    network = controller_gen.network
    # EDIT THE GENERIC CONTROLLER
    # Remove Head nodes
    network.remove_nodes_from(['joint_Head_flexion',
                               'joint_Head_extension',
                               'joint_HeadFake1_flexion',
                               'joint_HeadFake1_extension',
                               'joint_Proboscis_flexion',
                               'joint_Proboscis_extension',
                               'joint_Labellum_flexion',
                               'joint_Labellum_extension',
                               'joint_LAntenna_flexion',
                               'joint_LAntenna_extension',
                               'joint_RAntenna_flexion',
                               'joint_RAntenna_extension'])

    # Remove Abdomen nodes
    network.remove_nodes_from(['joint_A1A2_flexion',
                               'joint_A1A2_extension',
                               'joint_A3_flexion',
                               'joint_A3_extension',
                               'joint_A4_flexion',
                               'joint_A4_extension',
                               'joint_A5_flexion',
                               'joint_A5_extension',
                               'joint_A6_flexion',
                               'joint_A6_extension'])

    # Remove wings and haltere nodes
    for node in ['','Fake1','Fake2']:
        LwingNode = 'joint_LWing'+node
        RwingNode = 'joint_RWing'+node
        LhaltereNode = 'joint_LHaltere'+node
        RhaltereNode = 'joint_RHaltere'+node

        network.remove_nodes_from([LwingNode+'_flexion',
                                   RwingNode+'_flexion',
                                   LwingNode+'_extension',
                                   RwingNode+'_extension',
                                   LhaltereNode+'_flexion',
                                   RhaltereNode+'_flexion',
                                   LhaltereNode+'_extension',
                                   RhaltereNode+'_extension'])

    # Remove tarsi nodes
    for i in range(1,6):
        for tarsus in ['LF','LM','LH','RF','RM','RH']:
            tarsusNode = 'joint_'+tarsus+'Tarsus'+str(i)
            network.remove_nodes_from([tarsusNode+'_flexion',
                                       tarsusNode+'_extension'])

    # Remove Coxa extra DOF nodes
    for coxa in ['LF','LM','LH','RF','RM','RH']:
        coxaNode = 'joint_'+coxa+'Coxa'
        coxaFake1Node = 'joint_'+coxa+'CoxaFake1'
        coxaFake2Node = 'joint_'+coxa+'CoxaFake2'
        if 'F' in coxa:
            network.remove_nodes_from([coxaFake2Node+'_flexion',
                                       coxaFake2Node+'_extension'])
        else:
            network.remove_nodes_from([coxaNode+'_flexion',
                                       coxaNode+'_extension'])

        network.remove_nodes_from([coxaFake1Node+'_flexion',
                                       coxaFake1Node+'_extension'])



    # Connect limbs
    # AgnosticController.add_mutual_connection(
    #     network,
    #     'LHip_flexion',
    #     'RHip_flexion',
    #     weight=10.0,
    #     phi=np.pi
    # )

    with open('../config/network_node_positions.yaml', 'r') as file:
        node_positions = yaml.load(file, yaml.SafeLoader)
    for node, data in node_positions.items():
        network.nodes[node]['x'] = data[0]
        network.nodes[node]['y'] = data[1]
        network.nodes[node]['z'] = data[2]

    # EDIT CONNECTIONS FOR TRIPOD GAIT
    # Connecting base nodes
    weight = 10.0
    base_connections = [
        ['LFCoxa', 'RFCoxa', {'weight':weight, 'phi': np.pi}],
        ['LFCoxa', 'RMCoxaFake2', {'weight':weight, 'phi': 0.0}],
        ['RMCoxaFake2', 'LHCoxaFake2', {'weight':weight, 'phi': 0.0}],
        ['RFCoxa', 'LMCoxaFake2', {'weight':weight, 'phi': 0.0}],
        ['LMCoxaFake2', 'RHCoxaFake2', {'weight':weight, 'phi': 0.0}],
    ]

    for n1, n2, data in base_connections:
        AgnosticController.add_connection_antagonist(
            network,
            'joint_{}'.format(n1),
            'joint_{}'.format(n2),
            **data
        )

    leg_connections = [
        ['Coxa', 'Femur', {'weight':weight, 'phi': 0.0}],
        ['Femur', 'Tibia', {'weight':weight, 'phi': 0.0}],
    ]

    for n1, n2, data in leg_connections:
        for pos in ['F', 'M', 'H']:
            for side in ['L', 'R']:
                if (pos == 'M' or pos == 'H') and (n1 == 'Coxa'):
                    n1 = 'CoxaFake2'
                AgnosticController.add_connection_antagonist(
                    network,
                    'joint_{}{}{}'.format(side, pos, n1),
                    'joint_{}{}{}'.format(side, pos, n2),
                    **data
                )

    coxa_connections = [
        ['Coxa', 'Coxa', {'weight':weight, 'phi': np.pi/2}],
    ]

    for n1, n2, data in coxa_connections:
        for pos in ['F', 'M', 'H']:
            for side in ['L', 'R']:
                if (pos == 'M' or pos == 'H'):
                    n1 = 'CoxaFake2'
                    n2 = 'CoxaFake2'
                AgnosticController.add_mutual_connection(
                    network,
                    'joint_{}{}{}_{}'.format(side, pos, n1, 'flexion'),
                    'joint_{}{}{}_{}'.format(side, pos, n2, 'extension'),
                    **data
                )

    # for joint in controller_gen.model.joints:
    #     n1 = '{}_flexion'.format(joint.name)
    #     n2 = '{}_extension'.format(joint.name)
    #     network.remove_edges_from([(n1, n2), (n2, n1)])

    nx.write_graphml(network, net_dir)

    # Export position file to yaml
    # with open('../config/network_node_positions.yaml', 'w') as file:
    #     yaml.dump(node_positions, file, default_flow_style=True)

    # # Initialize network
    dt = 0.001  # Time step
    dur = 10
    time_vec = np.arange(0, dur, dt)  # Time
    container = Container(dur/dt)
    net = NeuralSystem(
        "../config/locomotion.graphml",
        container)

    # initialize network parameters
    container.initialize()
    net.setup_integrator()

    # Integrate the network
    pylog.info('Begin Integration!')

    for t in time_vec:
        net.step(dt=dt)
        container.update_log()

    # Results
    container.dump()
    state = np.asarray(container.neural.states.log)
    neuron_out = np.asarray(container.neural.outputs.log)
    names = container.neural.outputs.names
    parameters = container.neural.parameters

    # Show graph
    print(net.graph.number_of_edges())
    print(net.graph.number_of_nodes())
    net.visualize_network(edge_labels=False)
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
