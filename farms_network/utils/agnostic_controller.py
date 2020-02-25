""" Generate a template network. """

import farms_pylog as pylog
import networkx as nx
from farms_network.neural_system import NeuralSystem
from farms_sdf.sdf import ModelSDF
from farms_sdf import utils as sdf_utils
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container

pylog.set_level('debug')


class AgnosticController:
    """Generate agnostic muscle neural control.

    """

    def __init__(
            self,
            sdf_path,
            connect_mutual=True,
            connect_closest_neighbors=True,
            connect_base_nodes=True
    ):
        super().__init__()
        self.model = self.read_sdf(sdf_path)[0]
        self.connect_flexion_extension = connect_mutual
        self.connect_closest_neighbors = connect_closest_neighbors
        self.connect_base_nodes = connect_base_nodes
        #: Define a network graph
        self.network = nx.DiGraph()
        #: Generate the basic network
        self.generate_network()

    @staticmethod
    def read_sdf(sdf_path):
        """Read sdf model
        Keyword Arguments:
        sdf_path -- 
        """
        return ModelSDF.read(sdf_path)

    @staticmethod
    def add_mutual_connection(network, node_1, node_2, weight, phi):
        """
        Add mutual connection between two nodes
        """
        network.add_edge(
            node_1,
            node_2,
            weight=weight,
            phi=phi
        )
        network.add_edge(
            node_2,
            node_1,
            weight=weight,
            phi=-1*phi
        )

    @staticmethod
    def add_connection_to_closest_neighbors(network, model):
        """ Add connections to closest neighbors. """
        for joint in model.joints:
            for conn in sdf_utils.find_neighboring_joints(
                    model, joint.name):
                print("{} -> {}".format(joint.name, conn))
                AgnosticController.add_mutual_connection(
                    network,
                    joint.name + '_flexion',
                    conn + '_flexion',
                    weight=50.0,
                    phi=np.pi/2
                )
                AgnosticController.add_mutual_connection(
                    network,
                    joint.name + '_extension',
                    conn + '_extension',
                    weight=50.0,
                    phi=np.pi/2
                )

    @staticmethod
    def add_connection_between_base_nodes(network, model):
        """ Add connection between base nodes. """
        root_link = sdf_utils.find_root(model)
        base_joints = []
        for joint in model.joints:
            if joint.parent == root_link:
                base_joints.append(joint.name)
        for j1, j2 in itertools.combinations(base_joints, 2):
            AgnosticController.add_mutual_connection(
                network,
                j1 + '_flexion',
                j2 + '_flexion',
                weight=50.0,
                phi=0.0
            )
            AgnosticController.add_mutual_connection(
                network,
                j1 + '_extension',
                j2 + '_extension',
                weight=50.0,
                phi=0.0
            )

    def generate_network(self):
        """Generate network
        Keyword Arguments:
        self -- 
        """
        links = self.model.links
        link_id = sdf_utils.link_name_to_index(self.model)

        #: Add two neurons to each joint and connect each other
        for joint in self.model.joints:
            self.network.add_node(
                joint.name + '_flexion',
                model='oscillator',
                f=5,
                R=1.0,
                a=25,
                x=links[link_id[joint.child]].pose[0]+0.001,
                y=links[link_id[joint.child]].pose[1] +
                links[link_id[joint.child]].pose[2],
                z=links[link_id[joint.child]].pose[2],
            )
            self.network.add_node(
                joint.name + '_extension',
                model='oscillator',
                f=5,
                R=1.0,
                a=25,
                x=links[link_id[joint.child]].pose[0]-0.001,
                y=links[link_id[joint.child]].pose[1] +
                links[link_id[joint.child]].pose[2],
                z=links[link_id[joint.child]].pose[2],
            )
            if self.connect_flexion_extension:
                AgnosticController.add_mutual_connection(
                    self.network,
                    joint.name + '_flexion',
                    joint.name + '_extension',
                    weight=50.0,
                    phi=np.pi
                )

        #: Connect neurons to closest neighbors
        if self.connect_closest_neighbors:
            pylog.debug("Connecting closest neighbors")
            AgnosticController.add_connection_to_closest_neighbors(
                self.network,
                self.model
            )

        #: Connect neurons between the base nodes
        if self.connect_base_nodes:
            pylog.debug("Connecting base nodes")
            AgnosticController.add_connection_between_base_nodes(
                self.network,
                self.model
            )


def main():
    """ Main. """
    controller_gen = AgnosticController(
        ("../../../farms_blender/animats/"
         "mouse_v1/design/sdf/mouse_locomotion.sdf"),
    )
    net_dir = "../config/mouse_locomotion.graphml"
    nx.write_graphml(controller_gen.network, net_dir)

    # #: Initialize network
    dt = 0.001  #: Time step
    dur = 1
    time_vec = np.arange(0, dur, dt)  #: Time
    container = Container(dur/dt)
    net = NeuralSystem(
        "../config/mouse_locomotion.graphml",
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
