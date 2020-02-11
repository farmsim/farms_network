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
    
    def __init__(self, sdf_path):
        super().__init__()
        self.model = self.read_sdf(sdf_path)[0]
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
        
    def generate_network(self):
        """Generate network
        Keyword Arguments:
        self -- 
        """
        links  = self.model.links
        link_id = sdf_utils.link_name_to_index(self.model)
        joints = self.model.joints
        joint_id  = sdf_utils.joint_name_to_index(self.model)
        #: Add two neurons to each joint and connect each other
        for joint in joints:
            self.network.add_node(
                joint.name + '_flexion',
                model='oscillator',
                f=0.5,
                R=1.0,
                a=10.0,
                x=links[link_id[joint.child]].pose[0]+0.001,
                y=links[link_id[joint.child]].pose[1],
                z=links[link_id[joint.child]].pose[2],
            )
            self.network.add_node(
                joint.name + '_extension',
                model='oscillator',
                f=0.5,
                R=1.0,
                a=10.0,
                x=links[link_id[joint.child]].pose[0]-0.001,
                y=links[link_id[joint.child]].pose[1],
                z=links[link_id[joint.child]].pose[2],
            )
            AgnosticController.add_mutual_connection(
                self.network,
                joint.name + '_flexion',
                joint.name + '_extension',
                weight=10.0,
                phi=0.0
            )

        #: Connect neurons to closest neighbors
        for joint in joints:
            for conn in sdf_utils.find_neighboring_joints(
                    self.model, joint.name):
                print("{} -> {}".format(joint.name, conn))
                AgnosticController.add_mutual_connection(
                    self.network,
                    joint.name + '_flexion',
                    conn + '_flexion',
                    weight=10.0,
                    phi=0.0
                )
                AgnosticController.add_mutual_connection(
                    self.network,
                    joint.name + '_extension',
                    conn + '_extension',
                    weight=10.0,
                    phi=0.0
                )
        #: Connect base nodes
        root_link = sdf_utils.find_root(self.model)
        base_joints = []
        for joint in joints:
            if joint.parent == root_link:
                base_joints.append(joint.name)
        for j1, j2 in itertools.combinations(base_joints, 2):
            AgnosticController.add_mutual_connection(
                self.network,
                j1 + '_flexion',
                j2 + '_flexion',
                weight=10.0,
                    phi=0.0
            )
            AgnosticController.add_mutual_connection(
                self.network,
                j1 + '_extension',
                j2 + '_extension',
                weight=10.0,
                phi=0.0
            )

def main():
    """ Main. """
    controller_gen = AgnosticController(
        ("../../../farms_blender/animats/"
         "mouse_v1/design/sdf/mouse_locomotion.sdf")
    )    
    net_dir = "../config/temp.graphml"
    nx.write_graphml(controller_gen.network, net_dir)
    container = Container()
    net = NeuralSystem(
        "../config/temp.graphml",
        container)
    net.visualize_network(edge_labels=False)
    
    # nx.draw(controller_gen.network)
    # plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
