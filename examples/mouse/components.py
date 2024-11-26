""" Components """

import os
from pprint import pprint
from typing import Iterable, List

import farms_pylog as pylog
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from farms_network.core import options
from farms_core.io.yaml import read_yaml
from farms_network.core.data import NetworkData
from farms_network.core.network import PyNetwork
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

plt.rcParams["text.usetex"] = True


def calculate_arc_rad(source_pos, target_pos, base_rad=-0.1):
    """Calculate arc3 radius for edge based on node positions."""
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]

    # Set curvature to zero if nodes are aligned horizontally or vertically
    if dx == 0 or dy == 0:
        return 0.0

    # Decide on curvature based on position differences
    if abs(dx) > abs(dy):
        # Horizontal direction - positive rad for up, negative for down
        return -base_rad if dy >= 0 else base_rad
    else:
        # Vertical direction - positive rad for right, negative for left
        return base_rad if dx >= 0 else base_rad


def update_edge_visuals(network_options):
    """ Update edge options """

    nodes = network_options.nodes
    edges = network_options.edges
    for edge in edges:
        base_rad = calculate_arc_rad(
            nodes[nodes.index(edge.source)].visual.position,
            nodes[nodes.index(edge.target)].visual.position,
        )
        edge.visual.connectionstyle = f"arc3,rad={base_rad*0.0}"
    return network_options


def join_str(strings):
    return "_".join(filter(None, strings))


def multiply_transform(vec: np.ndarray, transform_mat: np.ndarray) -> np.ndarray:
    """
    Multiply a 2D vector with a 2D transformation matrix (3x3).

    Parameters:
    vec (np.ndarray): A 2D vector (shape (2,) or (3,))
    transform_mat (np.ndarray): A 3x3 transformation matrix.

    Returns:
    np.ndarray: The transformed vector.
    """

    assert transform_mat.shape == (3, 3), "Transformation matrix must be 3x3"

    # Ensure vec is in homogeneous coordinates (i.e., 3 elements).
    if vec.shape == (2,):
        vec = np.append(vec, 1)
    elif vec.shape != (3,):
        raise ValueError("Input vector must have shape (2,) or (3,)")

    # Perform the multiplication
    return transform_mat @ vec


def get_scale_matrix(scale: float) -> np.ndarray:
    """Return a scaling matrix."""
    return np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])


def get_mirror_matrix(mirror_x: bool = False, mirror_y: bool = False) -> np.ndarray:
    """Return a mirror matrix based on the mirror flags."""
    mirror_matrix = np.identity(3)
    if mirror_x:
        mirror_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    if mirror_y:
        mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return mirror_matrix


def get_translation_matrix(off_x: float, off_y: float) -> np.ndarray:
    """Return a translation matrix."""
    return np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]])


def get_rotation_matrix(angle: float) -> np.ndarray:
    """Return a rotation matrix for the given angle in degrees."""
    angle_rad = np.radians(angle)
    return np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )


def get_transform_mat(
    angle: float,
    off_x: float,
    off_y: float,
    mirror_x: bool = False,
    mirror_y: bool = False,
    scale: float = 2.5,
) -> np.ndarray:
    """Return a complete transformation matrix based on input parameters."""
    scale_matrix = get_scale_matrix(scale)
    mirror_matrix = get_mirror_matrix(mirror_x, mirror_y)
    translation_matrix = get_translation_matrix(off_x, off_y)
    rotation_matrix = get_rotation_matrix(angle)

    # Combine the transformations in the correct order: translation -> rotation -> mirror -> scale
    transform_matrix = translation_matrix @ rotation_matrix @ mirror_matrix
    transform_matrix = scale_matrix @ transform_matrix

    return transform_matrix


def create_node(
    base_name: str,
    node_id: str,
    node_type: str,
    position_vec: np.ndarray,
    label: str,
    color: list,
    transform_mat: np.ndarray,
    states: dict,
    parameters: dict,
) -> options.LIDannerNodeOptions:
    """
    Function to create a node with visual and state options.

    Parameters:
    base_name (str): The base name to prepend to node_id.
    node_id (str): Unique identifier for the node.
    position_vec (np.ndarray): The position of the node.
    label (str): The visual label for the node.
    color (list): RGB color values for the node.
    node_type (str): Type of the node ('LINaPDanner' or 'LIDanner').
    transform_mat (np.ndarray): Transformation matrix for positioning.
    v0 (float): Initial value for the state option 'v0'.
    h0 (float, optional): Initial value for the state option 'h0', only used for some node types.

    Returns:
    options.LIDannerNodeOptions: The configured node options object.
    """
    # Generate the full name and position
    full_name = join_str((base_name, node_id))
    position = multiply_transform(position_vec, transform_mat).tolist()

    # Determine node type and state options
    visual_options = options.NodeVisualOptions(
        position=position,
        label=label,
        color=color,
    )
    if node_type == "LINaPDanner":
        state_options = options.LINaPDannerStateOptions.from_kwargs(**states)
        parameters = options.LINaPDannerParameterOptions.defaults(**parameters)
        noise = options.OrnsteinUhlenbeckOptions.defaults()
        node_options_class = options.LINaPDannerNodeOptions
    elif node_type == "LIDanner":
        state_options = options.LIDannerStateOptions.from_kwargs(**states)
        parameters = options.LIDannerParameterOptions.defaults(**parameters)
        noise = options.OrnsteinUhlenbeckOptions.defaults()
        node_options_class = options.LIDannerNodeOptions
    elif node_type == "Linear":
        state_options = None
        parameters = options.LinearParameterOptions.defaults(**parameters)
        noise = None
        node_options_class = options.LinearNodeOptions
    elif node_type == "ExternalRelay":
        state_options = None
        parameters = options.NodeParameterOptions()
        noise = None
        visual_options.radius = 0.0
        node_options_class = options.ExternalRelayNodeOptions
    else:
        raise ValueError(f"Unknown node type: {node_type}")

    # Create and return the node options
    return node_options_class(
        name=full_name,
        parameters=parameters,
        visual=visual_options,
        state=state_options,
        noise=noise,
    )


def create_nodes(
    node_specs: Iterable,
    base_name: str,
    transform_mat: np.ndarray,
) -> options.NodeOptions:
    """Create node using create_method"""
    nodes = {}
    for (
        node_id,
        node_type,
        position_vec,
        label,
        color,
        states,
        parameters,
    ) in node_specs:
        nodes[node_id] = create_node(
            base_name,
            node_id,
            node_type,
            position_vec,
            label,
            color,
            transform_mat,
            states,
            parameters,
        )
    return nodes


def create_edges(
    edge_specs: Iterable,
    base_name: str,
    visual_options: options.EdgeVisualOptions = options.EdgeVisualOptions(),
) -> options.EdgeOptions:
    """Create edges from specs"""
    edges = {}
    for source_tuple, target_tuple, weight, edge_type in edge_specs:
        source = join_str((base_name, *source_tuple))
        target = join_str((base_name, *target_tuple))
        edges[join_str((source, "to", target))] = options.EdgeOptions(
            source=source,
            target=target,
            weight=weight,
            type=edge_type,
            visual=options.EdgeVisualOptions(**visual_options),
        )
    return edges


class RhythmGenerator:
    """Generate RhythmGenerator Network"""

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        super().__init__()
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""
        node_specs = [
            (
                join_str(("RG", "F")),
                "LINaPDanner",
                np.array((3.0, 0.0)),
                "F",
                [1.0, 0.0, 0.0],
                {"v": -62.5, "h": np.random.uniform(0, 1)},
                {},
            ),
            (
                join_str(("RG", "E")),
                "LINaPDanner",
                np.array((-3.0, 0.0)),
                "E",
                [0.0, 1.0, 0.0],
                {"v": -62.5, "h": np.random.uniform(0, 1)},
                {},
            ),
            (
                join_str(("RG", "In", "F")),
                "LIDanner",
                np.array((1.0, -1.5)),
                "In",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("RG", "In", "E")),
                "LIDanner",
                np.array((-1.0, 1.5)),
                "In",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("RG", "In", "E2")),
                "LIDanner",
                np.array((-5.0, 1.0)),
                "In",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("RG", "F", "DR")),
                "Linear",
                np.array((3.0, 2.0)),
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.1, "bias": 0.0},
            ),
            (
                join_str(("RG", "E", "DR")),
                "Linear",
                np.array((-3.0, 2.0)),
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.0, "bias": 0.1},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes

    def edges(self):
        """Add edges."""

        # Define edge details in a list for easier iteration
        edge_specs = [
            (("RG", "F"), ("RG", "In", "F"), 0.4, "excitatory"),
            (("RG", "In", "F"), ("RG", "E"), -1.0, "inhibitory"),
            (("RG", "E"), ("RG", "In", "E"), 0.4, "excitatory"),
            (("RG", "In", "E"), ("RG", "F"), -0.08, "inhibitory"),
            (("RG", "In", "E2"), ("RG", "F"), -0.04, "inhibitory"),
            (("RG", "F", "DR"), ("RG", "F"), 1.0, "excitatory"),
            (("RG", "E", "DR"), ("RG", "E"), 1.0, "excitatory"),
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges


class PatternFormation:
    """Generate PatternFormation Network"""

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        super().__init__()
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""
        nodes = {}

        node_specs = [
            (
                join_str(("PF", "FA")),
                "LINaPDanner",
                np.array((-3.0, 0.0)),
                "F\\textsubscript{A}",
                [1.0, 0.0, 0.0],
                {"v": -60.0, "h": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "EA")),
                "LINaPDanner",
                np.array((-9.0, 0.0)),
                "E\\textsubscript{A}",
                [0.0, 1.0, 0.0],
                {"v": -60.0, "h": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "In", "FA")),
                "LIDanner",
                np.array((-5.0, -1.5)),
                "In\\textsubscript{A}",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("PF", "In", "EA")),
                "LIDanner",
                np.array((-7.0, 1.5)),
                "In\\textsubscript{A}",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("PF", "FB")),
                "LINaPDanner",
                np.array((9.0, 0.0)),
                "F\\textsubscript{B}",
                [1.0, 0.0, 0.0],
                {"v": -60.0, "h": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "g_leak": 1.0, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "EB")),
                "LINaPDanner",
                np.array((3.0, 0.0)),
                "E\\textsubscript{B}",
                [0.0, 1.0, 0.0],
                {"v": -60.0, "h": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "g_leak": 1.0, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "In", "FB")),
                "LIDanner",
                np.array((7.0, -1.5)),
                "In\\textsubscript{B}",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("PF", "In", "EB")),
                "LIDanner",
                np.array((5.0, 1.5)),
                "In\\textsubscript{B}",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("PF", "In2", "F")),
                "LIDanner",
                np.array((9.0, -3.0)),
                "In\\textsubscript{2F}",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {"g_leak": 5.0},
            ),
            (
                join_str(("PF", "In2", "E")),
                "LIDanner",
                np.array((3.0, -3.0)),
                "In\\textsubscript{2E}",
                [0.2, 0.2, 0.2],
                {"v": -60.0},
                {"g_leak": 5.0},
            ),
            (
                join_str(("PF", "FA", "DR")),
                "Linear",
                np.array((-3.0, 2.0)),
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.6, "bias": 0.0},
            ),
            (
                join_str(("PF", "EA", "DR")),
                "Linear",
                np.array((-9.0, 2.0)),
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.6, "bias": 0.0},
            ),
            (
                join_str(("PF", "EB", "DR")),
                "Linear",
                np.array((3.0, 2.0)),
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.07, "bias": 0.0},
            ),
            (
                join_str(("PF", "In2", "E", "DR")),
                "Linear",
                np.array((4.0, -3.0)),
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.1, "bias": 0.0},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes

    def edges(self):
        """Add edges."""
        edges = {}

        # Define edge details in a list for easier iteration
        edge_specs = [
            (("PF", "FA"), ("PF", "In", "FA"), 0.8, "excitatory"),
            (("PF", "EA"), ("PF", "In", "EA"), 1.0, "excitatory"),
            (("PF", "In", "FA"), ("PF", "EA"), -1.5, "inhibitory"),
            (("PF", "In", "EA"), ("PF", "FA"), -1.0, "inhibitory"),

            (("PF", "FB"), ("PF", "In", "FB"), 1.5, "excitatory"),
            (("PF", "EB"), ("PF", "In", "EB"), 1.5, "excitatory"),
            (("PF", "In", "FB"), ("PF", "EB"), -2.0, "inhibitory"),
            (("PF", "In", "EB"), ("PF", "FB"), -0.25, "inhibitory"),

            (("PF", "In", "FA"), ("PF", "EB"), -0.5, "inhibitory"),
            (("PF", "In", "FA"), ("PF", "FB"), -0.1, "inhibitory"),
            (("PF", "In", "EA"), ("PF", "EB"), -0.5, "inhibitory"),
            (("PF", "In", "EA"), ("PF", "FB"), -0.25, "inhibitory"),

            (("PF", "In", "FB"), ("PF", "EA"), -0.5, "inhibitory"),
            (("PF", "In", "FB"), ("PF", "FA"), -0.75, "inhibitory"),
            (("PF", "In", "EB"), ("PF", "EA"), -2.0, "inhibitory"),
            (("PF", "In", "EB"), ("PF", "FA"), -2.0, "inhibitory"),

            (("PF", "In2", "F"), ("PF", "FB"), -3.0, "inhibitory"),
            (("PF", "In2", "E"), ("PF", "EB"), -3.0, "inhibitory"),

            (("PF", "FA", "DR"), ("PF", "FA"), 1.0, "excitatory"),
            (("PF", "EA", "DR"), ("PF", "EA"), 1.0, "excitatory"),
            (("PF", "EB", "DR"), ("PF", "EB"), 1.0, "excitatory"),
            (("PF", "In2", "E", "DR"), ("PF", "In2", "E"), -1.0, "inhibitory"),
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges


class Commissural:
    """Generate Commissural Network"""

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        super().__init__()
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""
        node_specs = [
            (
                "V2a",
                "LIDanner",
                np.array((0.0, 2.0, 1.0)),
                "V2\\textsubscript{a}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                "InV0V",
                "LIDanner",
                np.array((0.0, 0.0, 1.0)),
                "In\\textsubscript{i}",
                [1.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                "V0V",
                "LIDanner",
                np.array((2.0, 0.5, 1.0)),
                "V0\\textsubscript{V}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                "V0D",
                "LIDanner",
                np.array((2.0, -2.0, 1.0)),
                "V0\\textsubscript{D}",
                [1.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                "V3E",
                "LIDanner",
                np.array((2.0, 3.0, 1.0)),
                "V3\\textsubscript{E}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                "V3F",
                "LIDanner",
                np.array((2.0, -4.0, 1.0)),
                "V3\\textsubscript{F}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("V0V", "DR")),
                "Linear",
                np.array((3.0, 1.0)),
                "d",
                [0.5, 0.5, 0.5],
                None,
                {"slope": 0.15, "bias": 0.0},
            ),
            (
                join_str(("V0D", "DR")),
                "Linear",
                np.array((3.0, -2.5)),
                "d",
                [0.5, 0.5, 0.5],
                None,
                {"slope": 0.75, "bias": 0.0},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes

    def edges(self):
        """Add edges."""
        edges = {}

        # Define edge details in a list for easier iteration
        edge_specs = [
            (("V0V", "DR"), ("V0V",), -1.0, "inhibitory"),
            (("V0D", "DR"), ("V0D",), -1.0, "inhibitory"),
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges


class LPSN:
    """Generate Long Propriospinal Network"""

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        super().__init__()
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""

        # Define node specs in a list for easier iteration
        node_specs = [
            (
                join_str(("V0D", "diag")),
                "LIDanner",
                np.array((0.0, 0.0, 1.0)),
                "V0\\textsubscript{D}",
                [0.5, 0.0, 0.5],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("fore", "V0V", "diag")),
                "LIDanner",
                np.array((0.0, -1.25, 1.0)),
                "V0\\textsubscript{V}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("hind", "V3", "diag")),
                "LIDanner",
                np.array((0.0, -4.0, 1.0)),
                "V3\\textsubscript{a}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("fore", "Ini", "hom")),
                "LIDanner",
                np.array((-4.0, 0.0, 1.0)),
                "LPN\\textsubscript{i}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("fore", "Sh2", "hom")),
                "LIDanner",
                np.array((-8.0, 0.0, 1.0)),
                "Sh\\textsubscript{2}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("hind", "Sh2", "hom")),
                "LIDanner",
                np.array((-8.0, -4.0, 1.0)),
                "Sh\\textsubscript{2}",
                [0.0, 1.0, 0.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("V0D", "diag", "DR")),
                "Linear",
                np.array((1.0, 0.5)),
                "d",
                [0.5, 0.5, 0.5],
                None,
                {"slope": 0.75, "bias": 0.0},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes

    def edges(self):
        """Add edges."""
        edges = {}

        # Define edge details in a list for easier iteration
        edge_specs = [
            (("V0D", "diag", "DR"), ("V0D", "diag"), -1.0, "inhibitory")
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges


class MotorLayer:
    """Motorneurons and other associated interneurons."""

    def __init__(self, muscles: List[str], name="", transform_mat=np.identity(3)):
        """Initialization."""
        self.name = name
        self.muscles = muscles
        self.transform_mat = transform_mat

    def nodes(self):
        """Add neurons for the motor layer."""

        spacing = 2.5
        max_muscles = max(len(self.muscles["agonist"]), len(self.muscles["antagonist"]))

        node_specs = []
        # Define neurons for the muscles
        for x_off, muscle in zip(
            self._generate_positions(len(self.muscles["agonist"]), spacing),
            self.muscles["agonist"],
        ):
            node_specs.extend(self._get_muscle_neurons(muscle, x_off, 0.0))

        for x_off, muscle in zip(
            self._generate_positions(len(self.muscles["antagonist"]), spacing),
            self.muscles["antagonist"],
        ):
            node_specs.extend(
                self._get_muscle_neurons(muscle, x_off, 5.0, mirror_y=True)
            )

        # Calculate x positions for Ia inhibitory neurons
        IaIn_x_positions = np.linspace(-spacing * max_muscles, spacing * max_muscles, 4)
        y_off = 1.75

        node_specs += [
            (
                join_str(("Ia", "In", "EA")),
                "LIDanner",
                np.array((IaIn_x_positions[0], y_off, 1.0)),
                "Ia\\textsubscript{ea}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("Ia", "In", "EB")),
                "LIDanner",
                np.array((IaIn_x_positions[1], y_off, 1.0)),
                "Ia\\textsubscript{eb}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("Ia", "In", "FA")),
                "LIDanner",
                np.array((IaIn_x_positions[2], y_off, 1.0)),
                "Ia\\textsubscript{fa}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("Ia", "In", "FB")),
                "LIDanner",
                np.array((IaIn_x_positions[3], y_off, 1.0)),
                "Ia\\textsubscript{fb}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str(("Ib", "In", "RG")),
                "LIDanner",
                np.array((np.mean(IaIn_x_positions), y_off - spacing, 1.0)),
                "Ib\\textsubscript{rg}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes

    def edges(self):
        """Add motor feedback connections."""
        edges = {}

        # Define edges for each muscle
        for muscle in self.muscles["agonist"]:
            edges.update(self._generate_motor_connections(muscle))
        for muscle in self.muscles["antagonist"]:
            edges.update(self._generate_motor_connections(muscle))
        return edges

    def _generate_motor_connections(self, muscle):
        """Generate the motor connections for a specific muscle."""
        edge_specs = [
            *MotorLayer.connect_Rn_reciprocal_inhibition(muscle),
            *MotorLayer.connect_Ia_monosypatic_excitation(muscle),
            *MotorLayer.connect_Ib_disynaptic_inhibition(muscle),
        ]
        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges

    def _get_muscle_neurons(self, muscle, x_off, y_off, mirror_y=False):
        """Return neuron specifications for a muscle."""
        mirror_y_sign = -1 if mirror_y else 1
        return [
            (
                join_str((muscle["name"], "Mn")),
                "LIDanner",
                np.array((x_off, y_off, 1.0)),
                "Mn",
                [1.0, 0.0, 1.0],
                {"v": -60.0},
                {"e_leak": -52.5, "g_leak": 1.0},
            ),
            (
                join_str((muscle["name"], "Ia")),
                "ExternalRelay",
                np.array((x_off - 0.5, y_off + 0.75 * mirror_y_sign, 1.0)),
                "Ia",
                [1.0, 0.0, 0.0],
                {},
                {},
            ),
            (
                join_str((muscle["name"], "II")),
                "ExternalRelay",
                np.array((x_off, y_off + 0.75 * mirror_y_sign, 1.0)),
                "II",
                [1.0, 0.0, 0.0],
                {},
                {},
            ),
            (
                join_str((muscle["name"], "Ib")),
                "ExternalRelay",
                np.array((x_off + 0.5, y_off + 0.75 * mirror_y_sign, 1.0)),
                "Ib",
                [1.0, 0.0, 0.0],
                {},
                {},
            ),
            (
                join_str((muscle["name"], "Rn")),
                "LIDanner",
                np.array((x_off + 0.5, y_off - 1.0 * mirror_y_sign, 1.0)),
                "Rn",
                [1.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str((muscle["name"], "Ib", "In", "i")),
                "LIDanner",
                np.array((x_off + 1.0, y_off, 1.0)),
                "Ib\\textsubscript{i}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str((muscle["name"], "Ib", "In", "e")),
                "LIDanner",
                np.array((x_off + 1.0, y_off + 1.5 * mirror_y_sign, 1.0)),
                "Ib\\textsubscript{e}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
            (
                join_str((muscle["name"], "II", "In", "RG")),
                "LIDanner",
                np.array((x_off - 1.0, y_off, 1.0)),
                "II\\textsubscript{RG}",
                [0.0, 0.0, 1.0],
                {"v": -60.0},
                {},
            ),
        ]

    def _generate_positions(self, num_muscles, spacing):
        """Generate positions for the neurons."""
        return np.linspace(-spacing * num_muscles, spacing * num_muscles, num_muscles)

    @staticmethod
    def connect_Ia_monosypatic_excitation(muscle: str):
        edge_specs = [
            ((muscle["name"], "Ia"), (muscle["name"], "Mn"), 0.01, "excitatory"),
        ]
        return edge_specs

    @staticmethod
    def connect_Ib_disynaptic_inhibition(muscle: str):
        edge_specs = [
            ((muscle["name"], "Ib"), (muscle["name"], "Ib", "In", "i"), 0.01, "excitatory"),
            ((muscle["name"], "Ib", "In", "i"), (muscle["name"], "Mn"), -0.01, "inhibitory"),
        ]
        return edge_specs

    @staticmethod
    def connect_Rn_reciprocal_inhibition(muscle: str):
        """ Renshaw reciprocal inhibition """
        edge_specs = [
            ((muscle["name"], "Mn"), (muscle["name"], "Rn"), 0.01, "excitatory"),
            ((muscle["name"], "Rn"), (muscle["name"], "Mn"), -0.01, "inhibitory"),
        ]
        return edge_specs


######################
# CONNECT RG PATTERN #
######################
def connect_rhythm_pattern(base_name):
    """Connect RG's to pattern formation."""

    edge_specs = [
        (("RG", "F"), ("PF", "FA"), 0.8, "excitatory"),
        (("RG", "E"), ("PF", "EA"), 0.7, "excitatory"),

        (("RG", "F"), ("PF", "FB"), 0.6, "excitatory"),
        (("RG", "E"), ("PF", "EB"), 0.5, "excitatory"),

        (("RG", "F"), ("PF", "In2", "F"), 0.4, "excitatory"),
        (("RG", "E"), ("PF", "In2", "E"), 0.35, "excitatory"),

        (("RG", "In", "F"), ("PF", "EA"), -1.5, "inhibitory"),
        (("RG", "In", "E"), ("PF", "FA"), -1.5, "inhibitory"),
    ]

    # Use create_edges function to generate the edge options
    return create_edges(
        edge_specs,
        base_name=base_name,
        visual_options=options.EdgeVisualOptions()
    )


##########################
# Connect RG COMMISSURAL #
##########################
def connect_rg_commissural():
    """Connect RG's to Commissural."""

    edge_specs = []

    for limb in ("hind", "fore"):
        for side in ("left", "right"):
            edge_specs.extend([
                ((side, limb, "RG", "F"), (side, limb, "V2a"), 1.0, "excitatory"),
                ((side, limb, "RG", "F"), (side, limb, "V0D"), 0.7, "excitatory"),
                ((side, limb, "RG", "F"), (side, limb, "V3F"), 0.35, "excitatory"),
                ((side, limb, "RG", "E"), (side, limb, "V3E"), 0.35, "excitatory"),
                ((side, limb, "V2a"), (side, limb, "V0V"), 1.0, "excitatory"),
                ((side, limb, "InV0V"), (side, limb, "RG", "F"), -0.07, "inhibitory")
            ])

        # Handle cross-limb connections
        for sides in (("left", "right"), ("right", "left")):
            edge_specs.extend([
                ((sides[0], limb, "V0V"), (sides[1], limb, "InV0V"), 0.6, "excitatory"),
                ((sides[0], limb, "V0D"), (sides[1], limb, "RG", "F"), -0.07, "inhibitory"),
                ((sides[0], limb, "V3F"), (sides[1], limb, "RG", "F"), 0.03, "excitatory"),
                ((sides[0], limb, "V3E"), (sides[1], limb, "RG", "E"), 0.02, "excitatory"),
                ((sides[0], limb, "V3E"), (sides[1], limb, "RG", "In", "E"), 0.0, "excitatory"),
                ((sides[0], limb, "V3E"), (sides[1], limb, "RG", "In", "E2"), 0.8, "excitatory"),
            ])

    # Create the edges using create_edges
    edges = create_edges(
        edge_specs, base_name="", visual_options=options.EdgeVisualOptions()
    )
    return edges


##############################
# Connect Patter Commissural #
##############################
def connect_pattern_commissural():

    edge_specs = []

    for limb in ("hind",):
        for side in ("left", "right"):
            edge_specs.extend([
                ((side, limb, "V0D"), (side, limb, "PF", "FA"), -4.0, "inhibitory"),
                ((side, limb, "InV0V"), (side, limb, "PF", "FA"), -3.0, "inhibitory"),
            ])

    # Create the edges using create_edges
    edges = create_edges(
        edge_specs, base_name="", visual_options=options.EdgeVisualOptions()
    )
    return edges


def connect_fore_hind_circuits():
    """Connect CPG's to Interneurons."""

    edge_specs = []

    for side in ("left", "right"):
        edge_specs.extend([
            ((side, "fore", "RG", "F"), (side, "fore", "Ini", "hom"), 0.70, "excitatory"),
            ((side, "fore", "RG", "F"), (side, "V0D", "diag"), 0.50, "excitatory"),
            ((side, "fore", "Ini", "hom"), (side, "hind", "RG", "F"), -0.01, "inhibitory"),
            ((side, "fore", "Sh2", "hom"), (side, "hind", "RG", "F"), 0.01, "excitatory"),
            ((side, "hind", "Sh2", "hom"), (side, "fore", "RG", "F"), 0.05, "excitatory"),
            ((side, "fore", "RG", "F"), (side, "fore", "V0V", "diag"), 0.325, "excitatory"),
            ((side, "hind", "RG", "F"), (side, "hind", "V3", "diag"), 0.325, "excitatory")
        ])
        for limb in ("hind", "fore"):
            edge_specs.extend([
                ((side, limb, "RG", "E"), (side, limb, "Sh2", "hom"), 0.50, "excitatory")
            ])

    # Handle cross-limb connections
    for sides in (("left", "right"), ("right", "left")):
        edge_specs.extend([
            ((sides[0], "V0D", "diag"), (sides[1], "hind", "RG", "F"), -0.01, "inhibitory"),
            ((sides[0], "fore", "V0V", "diag"), (sides[1], "hind", "RG", "F"), 0.005, "excitatory"),
            ((sides[0], "hind", "V3", "diag"), (sides[1], "fore", "RG", "F"), 0.04, "excitatory")
        ])

    # Create the edges using create_edges
    edges = create_edges(
        edge_specs, base_name="", visual_options=options.EdgeVisualOptions()
    )

    return edges


def connect_pattern_motor_layer(base_name, muscle, patterns):
    """Return edge specs for connecting pattern formation to motor neuron layer."""
    edge_specs = [
        (("PF", pattern), (muscle, "Mn"), 0.1, "excitatory",)
        for pattern in patterns
    ]
    return create_edges(edge_specs, base_name=base_name)


def connect_pattern_to_IaIn(side, limb):
    """Return edge specs for connecting pattern formation to motor neuron layer."""
    edge_specs = [
        (("PF", pattern), ("Ia", "In", pattern), 0.01, "excitatory")
        for pattern in ("FA", "EA", "FB", "EB")
    ]
    return edge_specs


def connect_II_pattern_feedback(side, limb, muscle, patterns):
    """Return edge specs for connecting group II feedback to pattern layer."""
    edge_specs = [
        ((muscle, "II"), ("PF", pattern), 0.01, "excitatory")
        for pattern in patterns
    ]
    return edge_specs


def connect_Ia_reciprocal_inhibition_extensor2flexor(extensor: str, flexor: str):
    """Return edge specs for Ia reciprocal inhibition from extensor to flexor."""
    edge_specs = [
        ((extensor, "Ia"), ("Ia", "In", "EA"), 0.01, "excitatory"),
        ((extensor, "Ia"), ("Ia", "In", "EB"), 0.01, "excitatory"),
        (("Ia", "In", "EA"), (flexor, "Mn"), -0.01, "inhibitory"),
        (("Ia", "In", "EB"), (flexor, "Mn"), -0.01, "inhibitory"),
    ]
    return edge_specs


def connect_Ia_reciprocal_inhibition_flexor2extensor(flexor: str, extensor: str):
    """Return edge specs for Ia reciprocal inhibition from flexor to extensor."""
    edge_specs = [
        ((flexor, "Ia"), ("Ia", "In", "FA"), 0.01, "excitatory"),
        ((flexor, "Ia"), ("Ia", "In", "FB"), 0.01, "excitatory"),
        (("Ia", "In", "FA"), (extensor, "Mn"), -0.01, "inhibitory"),
        (("Ia", "In", "FB"), (extensor, "Mn"), -0.01, "inhibitory"),
    ]
    return edge_specs


def connect_Rn_reciprocal_facilitation_extensor2flexor(extensor: str, flexor: str):
    """Return edge specs for Rn reciprocal facilitation from extensor to flexor."""
    edge_specs = [
        ((extensor, "Rn"), ("Ia", "In", "EA"), -0.01, "inhibitory"),
        (("Ia", "In", "EA"), (flexor, "Mn"), -0.01, "inhibitory"),
        ((extensor, "Rn"), ("Ia", "In", "EB"), -0.01, "inhibitory"),
        (("Ia", "In", "EB"), (flexor, "Mn"), -0.01, "inhibitory"),
    ]
    return edge_specs


def connect_Rn_reciprocal_facilitation_flexor2extensor(flexor: str, extensor: str):
    """Return edge specs for Rn reciprocal facilitation from flexor to extensor."""
    edge_specs = [
        ((flexor, "Rn"), ("Ia", "In", "FA"), -0.01, "inhibitory"),
        (("Ia", "In", "FA"), (extensor, "Mn"), -0.01, "inhibitory"),
        ((flexor, "Rn"), ("Ia", "In", "FB"), -0.01, "inhibitory"),
        (("Ia", "In", "FB"), (extensor, "Mn"), -0.01, "inhibitory"),
    ]
    return edge_specs


def connect_Ib_disynaptic_extensor_excitation(extensor):
    """Return edge specs for Ib disynaptic excitation in extensor."""
    edge_specs = [
        ((extensor, "Ib"), (extensor, "Ib", "In", "e"), 1.0, "excitatory"),
        ((extensor, "Ib", "In", "e"), (extensor, "Mn"), 1.0, "excitatory"),
    ]
    return edge_specs


def connect_Ib_rg_feedback(extensor):
    """Return edge specs for Ib rhythm feedback."""
    edge_specs = [
        ((extensor, "Ib"), ("Ib", "In", "RG"), 1.0, "excitatory"),
        (("Ib", "In", "RG"), ("RG", "In", "E"), 1.0, "excitatory"),
        (("Ib", "In", "RG"), ("RG", "E"), 1.0, "excitatory"),
    ]
    return edge_specs


def connect_II_rg_feedback(flexor):
    """Return edge specs for II rhythm feedback."""
    edge_specs = [
        ((flexor, "II"), (flexor, "II", "In", "RG"), 1.0, "excitatory"),
        ((flexor, "II", "In", "RG"), ("RG", "F"), 1.0, "excitatory"),
        ((flexor, "II", "In", "RG"), ("RG", "In", "F"), 1.0, "excitatory"),
    ]
    return edge_specs


def connect_roll_vestibular_to_mn(side, fore_muscles, hind_muscles):
    """Return edge specs for roll vestibular to motor neurons."""
    rates = ("position", "velocity")
    weights = {"position": 0.01, "velocity": 0.01}
    edge_specs = []

    for rate in rates:
        for muscle in fore_muscles:
            name = "_".join(muscle["name"].split("_")[2:])
            edge_specs.append(
                (
                    (rate, "roll", "cclock", "In", "Vn"),
                    ("fore", name, "Mn"),
                    weights[rate],
                    "excitatory",
                )
            )

        for muscle in hind_muscles:
            name = "_".join(muscle["name"].split("_")[2:])
            edge_specs.append(
                (
                    (rate, "roll", "cclock", "In", "Vn"),
                    ("hind", name, "Mn"),
                    weights[rate],
                    "excitatory",
                )
            )

    return edge_specs


def connect_pitch_vestibular_to_mn(
    side, rate, fore_muscles, hind_muscles, default_weight=0.01
):
    """Return edge specs for pitch vestibular to motor neurons."""
    edge_specs = []

    for muscle in fore_muscles:
        name = "_".join(muscle["name"].split("_")[2:])
        edge_specs.append(
            (
                (rate, "pitch", "cclock", "In", "Vn"),
                ("fore", name, "Mn"),
                default_weight,
                "excitatory",
            )
        )

    for muscle in hind_muscles:
        name = "_".join(muscle["name"].split("_")[2:])
        edge_specs.append(
            (
                (rate, "pitch", "clock", "In", "Vn"),
                ("hind", name, "Mn"),
                default_weight,
                "excitatory",
            )
        )

    return edge_specs


###########
# Muscles #
###########
def define_muscle_patterns() -> dict:
    muscles_patterns = {
        "hind": {
            "bfa": ["EA", "EB"],
            "ip": ["FA", "FB"],
            "bfpst": ["FA", "EA", "FB", "EB"],
            "rf": ["EA", "FB", "EB"],
            "va": ["EA", "FB", "EB"],
            "mg": ["FA", "EA", "EB"],
            "sol": ["EA", "EB"],
            "ta": ["FA", "FB"],
            "ab": ["FA", "EA", "FB", "EB"],
            "gm_dorsal": ["FA", "EA", "FB", "EB"],
            "edl": ["FA", "EA", "FB", "EB"],
            "fdl": ["FA", "EA", "FB", "EB"],
        },
        "fore": {
            "spd": ["FA", "EA", "FB", "EB"],
            "ssp": ["FA", "EA", "FB", "EB"],
            "abd": ["FA", "EA", "FB", "EB"],
            "add": ["FA", "EA", "FB", "EB"],
            "tbl": ["FA", "EA", "FB", "EB"],
            "tbo": ["FA", "EA", "FB", "EB"],
            "bbs": ["FA", "FB"],
            "bra": ["FA", "EA", "FB", "EB"],
            "ecu": ["FA", "EA", "FB", "EB"],
            "fcu": ["FA", "EA", "FB", "EB"],
        }
    }
    return muscles_patterns


def generate_muscle_agonist_antagonist_pairs(muscle_config_path: str) -> dict:
    # read muscle config file
    muscles_config = read_yaml(muscle_config_path)

    sides = ("left", "right")
    limbs = ("hind", "fore")

    muscles = {
        sides[0]: {
            limbs[0]: {"agonist": [], "antagonist": []},
            limbs[1]: {"agonist": [], "antagonist": []},
        },
        sides[1]: {
            limbs[0]: {"agonist": [], "antagonist": []},
            limbs[1]: {"agonist": [], "antagonist": []},
        },
    }

    for _name, muscle in muscles_config["muscles"].items():
        _side = muscle["side"]
        _limb = muscle["limb"]
        function = muscle.get("function", "agonist")
        muscles[_side][_limb][function].append(
            {
                "name": join_str(_name.split("_")[2:]),
                "type": muscle["type"],
                "abbrev": muscle["abbrev"],
            }
        )
    return muscles


################
# Limb Circuit #
################
def limb_circuit(
        network_options: options.NetworkOptions,
        side: str,
        limb: str,
        transform_mat=np.identity(3)
):

    # TODO: Change the use of side and limb attr name in loops
    # Base name
    name = join_str((side, limb))

    #####################
    # Rhythm generation #
    #####################
    rhythm = RhythmGenerator(name=name, transform_mat=transform_mat)
    network_options.add_nodes(rhythm.nodes().values())
    network_options.add_edges(rhythm.edges().values())

    #####################
    # Pattern Formation #
    #####################
    pattern_transformation_mat = transform_mat@(
        get_translation_matrix(off_x=0.0, off_y=7.5)
    )

    pattern = PatternFormation(
        name=name,
        transform_mat=pattern_transformation_mat
    )
    network_options.add_nodes(pattern.nodes().values())
    network_options.add_edges(pattern.edges().values())

    # Connect sub layers
    rhythm_pattern_edges = connect_rhythm_pattern(name)
    network_options.add_edges(rhythm_pattern_edges.values())


    ##############
    # MotorLayer #
    ##############
    # read muscle config file
    muscles_config_path = "/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/muscles/quadruped_siggraph.yaml"
    muscles = generate_muscle_agonist_antagonist_pairs(muscles_config_path)

    ###################################
    # Connect patterns and motorlayer #
    ###################################
    muscles_patterns = define_muscle_patterns()

    motor_transformation_mat = pattern_transformation_mat@(
        get_translation_matrix(
            off_x=0.0,
            off_y=5.0
        )
    )

    motor = MotorLayer(
        muscles=muscles[side][limb],
        name=name,
        transform_mat=motor_transformation_mat
    )

    network_options.add_nodes(motor.nodes().values())
    network_options.add_edges(motor.edges().values())

    # Connect pattern formation to motor neurons
    for muscle, patterns in muscles_patterns[limb].items():
        pattern_motor_edges = connect_pattern_motor_layer(name, muscle, patterns)
        network_options.add_edges(pattern_motor_edges.values())

    # Connect pattern formation to IaIn
    edge_specs = connect_pattern_to_IaIn(side, limb)
    network_options.add_edges((create_edges(edge_specs, name)).values())

    #####################
    # Connect afferents #
    #####################
    edge_specs = []
    Ib_feedback_to_rg = {
        "hind": ["mg", "sol", "fdl"],
        "fore": ["fcu"]
    }

    Ib_feedback_disynaptic_excitation ={
        "hind": ["bfa", "bfpst", "va", "rf", "mg", "sol", "fdl"],
        "fore": ["ssp", "tbl", "tbo", "fcu"]
    }

    II_feedback_to_pattern = {
        "hind": {
            "ip": muscles_patterns["hind"]["ip"],
            "ta": muscles_patterns["hind"]["ta"],
            "edl": muscles_patterns["hind"]["edl"],
        },
        "fore": {
            "ssp": muscles_patterns["fore"]["ssp"],
            "bra": muscles_patterns["fore"]["bra"],
        }
    }

    II_feedback_to_rg = {
        "hind": ["ip", "ta",],
        "fore": ["ssp", "bra"]
    }

    Ia_reciprocal_inhibition_extensor2flexor = {
        "hind": {
            "extensors": ["bfa", "bfpst", "rf", "va", "mg", "sol", "fdl"],
            "flexors": ["ip", "ta", "edl"],
        },
        "fore": {
            "extensors": ["ssp", "tbl", "tbo", "fcu"],
            "flexors": ["spd", "bra", "bbs", "ecu"],
        }
    }

    Ia_reciprocal_inhibition_flexor2extensor = {
        "hind": {
            "extensors": ["bfa", "bfpst", "rf", "va", "mg", "sol", "fdl"],
            "flexors": ["ip", "ta", "edl"],
        },
        "fore": {
            "extensors": ["ssp", "tbl", "tbo", "fcu"],
            "flexors": ["spd", "bra", "bbs", "ecu"],
        }
    }

    renshaw_reciprocal_facilitation_extensor2flexor = {
        "hind": {
            "extensors": ["bfa", "bfpst", "rf", "va", "mg", "sol", "fdl"],
            "flexors": ["ip", "ta", "edl"],
        },
        "fore": {
            "extensors": ["ssp", "tbl", "tbo", "fcu"],
            "flexors": ["spd", "bra", "bbs", "ecu"],
        }
    }

    renshaw_reciprocal_facilitation_flexor2extensor = {
        "hind": {
            "extensors": ["bfa", "bfpst", "rf", "va", "mg", "sol", "fdl"],
            "flexors": ["ip", "ta", "edl"],
        },
        "fore": {
            "extensors": ["ssp", "tbl", "tbo", "fcu"],
            "flexors": ["spd", "bra", "bbs", "ecu"],
        }
    }
    # Type II connections
    # II to Pattern
    for muscle, patterns in II_feedback_to_pattern[limb].items():
        edge_specs += connect_II_pattern_feedback(
            side, limb=limb, muscle=muscle, patterns=patterns
        )

    for flexor in II_feedback_to_rg[limb]:
        edge_specs += connect_II_rg_feedback(flexor)
    # Type Ib connections
    # Ib to RG
    for extensor in Ib_feedback_to_rg[limb]:
        edge_specs += connect_Ib_rg_feedback(extensor)
    # Ib Disynaptic extensor excitation
    for extensor in Ib_feedback_disynaptic_excitation[limb]:
        edge_specs += connect_Ib_disynaptic_extensor_excitation(extensor)
    # Type Ia connections
    # Ia reciprocal inhibition extensor to flexor
    for extensor in Ia_reciprocal_inhibition_extensor2flexor[limb]["extensors"]:
        for flexor in Ia_reciprocal_inhibition_extensor2flexor[limb]["flexors"]:
            edge_specs += connect_Ia_reciprocal_inhibition_extensor2flexor(
                extensor, flexor
            )
    # Ia reciprocal inhibition flexor to extensor
    for flexor in Ia_reciprocal_inhibition_flexor2extensor[limb]["flexors"]:
        for extensor in Ia_reciprocal_inhibition_flexor2extensor[limb]["extensors"]:
            edge_specs += connect_Ia_reciprocal_inhibition_flexor2extensor(
                flexor, extensor
            )
    # Renshaw recurrent connections
    # renshaw reciprocal facilitation extensor to flexor
    for extensor in renshaw_reciprocal_facilitation_extensor2flexor[limb]["extensors"]:
        for flexor in renshaw_reciprocal_facilitation_extensor2flexor[limb]["flexors"]:
            edge_specs += connect_Rn_reciprocal_facilitation_extensor2flexor(extensor, flexor)
    # renshaw reciprocal facilitation flexor to extensor
    for flexor in renshaw_reciprocal_facilitation_flexor2extensor[limb]["flexors"]:
        for extensor in renshaw_reciprocal_facilitation_flexor2extensor[limb]["extensors"]:
            edge_specs += connect_Rn_reciprocal_facilitation_flexor2extensor(flexor, extensor)

    edges = create_edges(edge_specs, name)
    network_options.add_edges(edges.values())
    return network_options


#####################
# Interlimb Circuit #
#####################
def interlimb_circuit(
        network_options: options.NetworkOptions,
        sides: List[str],
        limbs: List[str],
        transform_mat=np.identity(3)
):
    for side in sides:
        for limb in limbs:
            commissural_offset_x, commissural_offset_y = 5.0, 2.5  # Independent offsets
            off_x = -commissural_offset_x if side == "left" else commissural_offset_x
            off_y = (commissural_offset_y + 20) if limb == "fore" else commissural_offset_y
            mirror_x = limb == "hind"
            mirror_y = side == "right"
            commissural = Commissural(
                name=join_str((side, limb)), transform_mat=(
                    transform_mat@get_translation_matrix(off_x=off_x, off_y=off_y)@get_mirror_matrix(
                        mirror_x=mirror_y, mirror_y=mirror_y
                    )
                )
            )
            network_options.add_nodes(commissural.nodes().values())
            network_options.add_edges(commissural.edges().values())

        lpsn_x_offset = 25.0
        lpsn_y_offset = 20 - 5.5  # Adjusted relative to base fore_y_offset
        off_x = -lpsn_x_offset if side == "left" else lpsn_x_offset
        off_y = lpsn_y_offset
        mirror_y = side == "right"
        lpsn = LPSN(
            name=side,
            transform_mat=(
                    transform_mat@get_translation_matrix(off_x=off_x, off_y=off_y)@get_mirror_matrix(
                        mirror_x=False, mirror_y=mirror_y
                    )
                )
        )
        network_options.add_nodes(lpsn.nodes().values())
        network_options.add_edges(lpsn.edges().values())

    return network_options


#####################
# Quadruped Circuit #
#####################
def quadruped_circuit(
        network_options: options.NetworkOptions,
        transform_mat=np.identity(3)
):
    """ Full Quadruped Circuit """

    # Limb circuitry
    network_options = limb_circuit(
        network_options,
        side="left",
        limb="hind",
        transform_mat=get_translation_matrix(
            off_x=-25.0, off_y=0.0
        )@get_mirror_matrix(
            mirror_x=True, mirror_y=False
        )@get_rotation_matrix(angle=45)
    )

    network_options = limb_circuit(
        network_options,
        side="right",
        limb="hind",
        transform_mat=get_translation_matrix(
            off_x=25.0, off_y=0.0
        )@get_mirror_matrix(
            mirror_x=True, mirror_y=False
        )@get_rotation_matrix(angle=-45)
    )

    network_options = limb_circuit(
        network_options,
        side="left",
        limb="fore",
        transform_mat=get_translation_matrix(
            off_x=-25.0, off_y=25.0
        )@get_rotation_matrix(angle=45)
    )

    network_options = limb_circuit(
        network_options,
        side="right",
        limb="fore",
        transform_mat=get_translation_matrix(
            off_x=25.0, off_y=25.0
        )@get_rotation_matrix(angle=-45)
    )

    # Commisural
    network_options = interlimb_circuit(
        network_options,
        sides=("left", "right"),
        limbs=("hind", "fore",),
    )

    #################################
    # Connect rhythm to commissural #
    #################################
    rg_commissural_edges = connect_rg_commissural()
    network_options.add_edges(rg_commissural_edges.values())

    ##################################
    # Connect pattern to commissural #
    ##################################
    pattern_commissural_edges = connect_pattern_commissural()
    network_options.add_edges(pattern_commissural_edges.values())

    ##############################
    # Connect fore and hind lpsn #
    ##############################
    fore_hind_edges = connect_fore_hind_circuits()
    network_options.add_edges(fore_hind_edges.values())

    return network_options
