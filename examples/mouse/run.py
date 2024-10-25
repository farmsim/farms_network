""" Generate and reproduce Zhang, Shevtsova, et al. eLife 2022;11:e73424. DOI:
https://doi.org/10.7554/eLife.73424 paper network """

import os
from pprint import pprint
from typing import List, Iterable

import farms_pylog as pylog
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from farms_network.core import options
from farms_network.core.data import NetworkData
from farms_network.core.network import PyNetwork, rk4
# from farms_network.gui.gui import NetworkGUI
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

plt.rcParams["text.usetex"] = True


def join_str(strings):
    return "_".join(strings)


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


def get_mirror_matrix(mirror_x: bool, mirror_y: bool) -> np.ndarray:
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
    if node_type == "LINaPDanner":
        state_options = options.LINaPDannerStateOptions.from_kwargs(**states)
        parameters = options.LINaPDannerParameterOptions.defaults(**parameters)
        node_options_class = options.LINaPDannerNodeOptions
    elif node_type == "LIDanner":
        state_options = options.LIDannerStateOptions.from_kwargs(**states)
        parameters = options.LIDannerParameterOptions.defaults(**parameters)
        node_options_class = options.LIDannerNodeOptions
    elif node_type == "Linear":
        state_options = None
        parameters = options.LinearParameterOptions.defaults(**parameters)
        node_options_class = options.LinearNodeOptions
    else:
        raise ValueError(f"Unknown node type: {node_type}")

    # Create and return the node options
    return node_options_class(
        name=full_name,
        parameters=parameters,
        visual=options.NodeVisualOptions(position=position, label=label, color=color),
        state=state_options,
    )


def create_nodes(
        node_specs: Iterable,
        base_name: str,
        transform_mat: np.ndarray,
) -> options.NodeOptions:
    """ Create node using create_method """
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
    """ Create edges from specs"""
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
                {"v0": -62.5, "h0": np.random.uniform(0, 1)},
                {},
            ),
            (
                join_str(("RG", "E")),
                "LINaPDanner",
                np.array((-3.0, 0.0)),
                "E",
                [0.0, 1.0, 0.0],
                {"v0": -62.5, "h0": np.random.uniform(0, 1)},
                {},
            ),
            (
                join_str(("RG", "In", "F")),
                "LIDanner",
                np.array((1.0, -1.5)),
                "In",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("RG", "In", "E")),
                "LIDanner",
                np.array((-1.0, 1.5)),
                "In",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("RG", "In", "E2")),
                "LIDanner",
                np.array((-5.0, 1.0)),
                "In",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {},
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
            (("RG", "In", "E2"), ("RG", "F"), -0.04, "excitatory"),
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges


class RhythmDrive:
    """Generate Drive Network"""

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""
        nodes = {}

        node_specs = [
            (
                join_str(("RG", "F", "DR")),
                "Linear",
                np.array((3.0, 2.0)),  # Assuming position is not important
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.1, "bias": 0.0},
            ),
            (
                join_str(("RG", "E", "DR")),
                "Linear",
                np.array((-3.0, 2.0)),  # Assuming position is not important
                "d",
                [0.5, 0.5, 0.5],  # Default visual color if needed
                None,
                {"slope": 0.0, "bias": 0.1},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes


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
                {"v0": -60.0, "h0": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "EA")),
                "LINaPDanner",
                np.array((-9.0, 0.0)),
                "F\\textsubscript{A}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0, "h0": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "In", "FA")),
                "LIDanner",
                np.array((-5.0, -1.5)),
                "In\\textsubscript{A}",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("PF", "In", "EA")),
                "LIDanner",
                np.array((-7.0, 1.5)),
                "In\\textsubscript{A}",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("PF", "FB")),
                "LINaPDanner",
                np.array((9.0, 0.0)),
                "F\\textsubscript{A}",
                [1.0, 0.0, 0.0],
                {"v0": -60.0, "h0": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "EB")),
                "LINaPDanner",
                np.array((3.0, 0.0)),
                "F\\textsubscript{A}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0, "h0": np.random.uniform(0, 1)},
                {"g_nap": 0.125, "e_leak": -67.5},
            ),
            (
                join_str(("PF", "In", "FB")),
                "LIDanner",
                np.array((7.0, -1.5)),
                "In\\textsubscript{B}",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("PF", "In", "EB")),
                "LIDanner",
                np.array((5.0, 1.5)),
                "In\\textsubscript{B}",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("PF", "In2", "F")),
                "LIDanner",
                np.array((9.0, -3.0)),
                "In\\textsubscript{2F}",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {"g_leak": 5.0},
            ),
            (
                join_str(("PF", "In2", "E")),
                "LIDanner",
                np.array((3.0, -3.0)),
                "In\\textsubscript{2E}",
                [0.2, 0.2, 0.2],
                {"v0": -60.0},
                {"g_leak": 5.0},
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
            (("PF", "In", "FA"), ("PF", "EA"), -1.5, "inhibitory"),
            (("PF", "EA"), ("PF", "In", "EA"), 1.0, "excitatory"),
            (("PF", "In", "EA"), ("PF", "FA"), -1.0, "inhibitory"),
            (("PF", "FB"), ("PF", "In", "FB"), 1.5, "excitatory"),
            (("PF", "In", "FB"), ("PF", "EB"), -2.0, "inhibitory"),
            (("PF", "EB"), ("PF", "In", "EB"), 1.5, "excitatory"),
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
                {"v0": -60.0},
                {},
            ),
            (
                "InV0V",
                "LIDanner",
                np.array((0.0, 0.0, 1.0)),
                "In\\textsubscript{i}",
                [1.0, 0.0, 1.0],
                {"v0": -60.0},
                {},
            ),
            (
                "V0V",
                "LIDanner",
                np.array((2.0, 0.5, 1.0)),
                "V0\\textsubscript{V}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
            (
                "V0D",
                "LIDanner",
                np.array((2.0, -2.0, 1.0)),
                "V0\\textsubscript{D}",
                [1.0, 0.0, 1.0],
                {"v0": -60.0},
                {},
            ),
            (
                "V3E",
                "LIDanner",
                np.array((2.0, 3.0, 1.0)),
                "V3\\textsubscript{E}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
            (
                "V3F",
                "LIDanner",
                np.array((2.0, -4.0, 1.0)),
                "V3\\textsubscript{F}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes


class CommissuralDrive:
    """Generate Drive Network"""

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""

        node_specs = [
            (
                join_str(("V0V", "DR")),
                "Linear",
                np.array((0.0, 0.0)),
                "d",
                [0.5, 0.5, 0.5],
                None,
                {"slope": 0.15, "bias": 0.0},
            ),
            (
                join_str(("V0D", "DR")),
                "Linear",
                np.array((0.0, 0.0)),
                "d",
                [0.5, 0.5, 0.5],
                None,
                {"slope": 0.75, "bias": 0.0},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes


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
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("fore", "V0V", "diag")),
                "LIDanner",
                np.array((0.0, -1.25, 1.0)),
                "V0\\textsubscript{V}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("hind", "V3", "diag")),
                "LIDanner",
                np.array((0.0, -4.0, 1.0)),
                "V3\\textsubscript{a}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("fore", "Ini", "hom")),
                "LIDanner",
                np.array((-4.0, 0.0, 1.0)),
                "LPN\\textsubscript{i}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("fore", "Sh2", "hom")),
                "LIDanner",
                np.array((-8.0, 0.0, 1.0)),
                "Sh\\textsubscript{2}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
            (
                join_str(("hind", "Sh2", "hom")),
                "LIDanner",
                np.array((-8.0, -4.0, 1.0)),
                "Sh\\textsubscript{2}",
                [0.0, 1.0, 0.0],
                {"v0": -60.0},
                {},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes


class LPSNDrive:
    """Generate Drive Network"""

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""

        node_specs = [
            (
                join_str(("V0D", "diag", "DR")),
                "Linear",
                np.array((0.0, 0.0)),
                "d",
                [0.5, 0.5, 0.5],
                None,
                {"slope": 0.75, "bias": 0.0},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes


class MotorLayer:
    """Motorneurons and other associated interneurons."""

    def __init__(self, muscles: List[str], name="", transform_mat=np.identity(3)):
        """Initialization."""
        self.name = name
        self.muscles = muscles
        self.transform_mat = transform_mat

    def nodes(self):
        """Add neurons for the motor layer."""

        neuron_specs = []

        # Define neurons for the muscles
        for x_off, muscle in zip(self._generate_positions(len(self.muscles["agonist"])), self.muscles["agonist"]):
            neuron_specs.extend(self._get_muscle_neurons(muscle, x_off, 0.0))

        for x_off, muscle in zip(self._generate_positions(len(self.muscles["antagonist"])), self.muscles["antagonist"]):
            neuron_specs.extend(self._get_muscle_neurons(muscle, x_off, 3.5, mirror_y=True))

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
        edges = {}

        edge_specs = [
            (f"{muscle}_Ia", f"{muscle}_Mn", 0.01, "excitatory", "Ia_monosynaptic_excitation"),
            (f"{muscle}_Mn", f"{muscle}_Rn", 0.01, "excitatory", "Rn_reciprocal_inhibition"),
            (f"{muscle}_Rn", f"{muscle}_Mn", -0.01, "inhibitory", "Rn_reciprocal_inhibition"),
            (f"{muscle}_Ib", f"{muscle}_IbIn_i", 0.01, "excitatory", "Ib_disynaptic_inhibition"),
            (f"{muscle}_IbIn_i", f"{muscle}_Mn", -0.01, "inhibitory", "Ib_disynaptic_inhibition"),
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges

    def _get_muscle_neurons(self, muscle, x_off, y_off, mirror_y=False):
        """Return neuron specifications for a muscle."""
        mirror_y_sign = -1 if mirror_y else 1

        return [
            (
                f"{muscle}_Mn", "LIDanner", (x_off, y_off, 1.0), "Mn", [1.0, 0.0, 1.0],
                {"v0": -60.0}, {"e_leak": -52.5, "g_leak": 1.0}
            ),
            (
                f"{muscle}_Ia", "LIDanner", (x_off - 0.5, y_off + 0.75 * mirror_y_sign, 1.0), "Ia", [1.0, 0.0, 0.0],
                {"init": 0.0}, {}
            ),
            (
                f"{muscle}_II", "LIDanner", (x_off, y_off + 0.75 * mirror_y_sign, 1.0), "II", [1.0, 0.0, 0.0],
                {"init": 0.0}, {}
            ),
            (
                f"{muscle}_Ib", "LIDanner", (x_off + 0.5, y_off + 0.75 * mirror_y_sign, 1.0), "Ib", [1.0, 0.0, 0.0],
                {"init": 0.0}, {}
            ),
            (
                f"{muscle}_Rn", "LIDanner", (x_off + 0.5, y_off - 1.0 * mirror_y_sign, 1.0), "Rn", [1.0, 0.0, 1.0],
                {"v0": -60.0}, {}
            ),
            (
                f"{muscle}_IbIn_i", "LIDanner", (x_off + 1.0, y_off, 1.0), "Ib\\textsubscript{i}", [0.0, 0.0, 1.0],
                {"v0": -60.0}, {}
            ),
            (
                f"{muscle}_IbIn_e", "LIDanner", (x_off + 1.0, y_off + 1.5 * mirror_y_sign, 1.0), "Ib\\textsubscript{e}", [0.0, 0.0, 1.0],
                {"v0": -60.0}, {}
            ),
            (
                f"{muscle}_IIIn_RG", "LIDanner", (x_off - 1.0, y_off, 1.0), "II\\textsubscript{RG}", [0.0, 0.0, 1.0],
                {"v0": -60.0}, {}
            ),
        ]

    def _generate_positions(self, num_muscles):
        """Generate positions for the neurons."""
        spacing = 1.25
        return np.linspace(-spacing * num_muscles, spacing * num_muscles, num_muscles)


##########################
# CONNECT RG COMMISSURAL #
##########################
def connect_rg_commissural():
    """Connect RG's to Commissural."""

    edges = {}
    for limb in ("hind", "fore"):
        for side in ("left", "right"):

            edges[join_str((side, limb, "RG", "F", "to", side, limb, "V2a"))] = (
                options.EdgeOptions(
                    source=join_str((side, limb, "RG", "F")),
                    target=join_str((side, limb, "V2a")),
                    weight=1.0,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )

            # edges[join_str((side, limb, "RG" "F", "to", side, limb, "V2a", "diag"))] = options.EdgeOptions(
            #     source=join_str((side, limb, "RG" "F")),
            #     target=join_str((side, limb, "V2a", "diag")),
            #     weight=0.5,
            #     type="excitatory",
            #     visual=options.EdgeVisualOptions(),

            # )
            edges[join_str((side, limb, "RG", "F", "to", side, limb, "V0D"))] = (
                options.EdgeOptions(
                    source=join_str((side, limb, "RG", "F")),
                    target=join_str((side, limb, "V0D")),
                    weight=0.7,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )

            edges[join_str((side, limb, "RG", "F", "to", side, limb, "V3F"))] = (
                options.EdgeOptions(
                    source=join_str((side, limb, "RG", "F")),
                    target=join_str((side, limb, "V3F")),
                    weight=0.35,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )

            edges[join_str((side, limb, "RG", "E", "to", side, limb, "V3E"))] = (
                options.EdgeOptions(
                    source=join_str((side, limb, "RG", "E")),
                    target=join_str((side, limb, "V3E")),
                    weight=0.35,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )

            edges[join_str((side, limb, "V2a", "to", side, limb, "V0V"))] = (
                options.EdgeOptions(
                    source=join_str((side, limb, "V2a")),
                    target=join_str((side, limb, "V0V")),
                    weight=1.0,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )

            edges[join_str((side, limb, "InV0V", "to", side, limb, "RG", "F"))] = (
                options.EdgeOptions(
                    source=join_str((side, limb, "InV0V")),
                    target=join_str((side, limb, "RG", "F")),
                    weight=-0.07,
                    type="inhibitory",
                    visual=options.EdgeVisualOptions(),
                )
            )

        for sides in (("left", "right"), ("right", "left")):

            edges[join_str((sides[0], limb, "V0V", "to", sides[1], limb, "InV0V"))] = (
                options.EdgeOptions(
                    source=join_str((sides[0], limb, "V0V")),
                    target=join_str((sides[1], limb, "InV0V")),
                    weight=0.6,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )

            edges[
                join_str((sides[0], limb, "V0D", "to", sides[1], limb, "RG", "F"))
            ] = options.EdgeOptions(
                source=join_str((sides[0], limb, "V0D")),
                target=join_str((sides[1], limb, "RG", "F")),
                weight=-0.07,
                type="inhibitory",
                visual=options.EdgeVisualOptions(),
            )

            edges[
                join_str((sides[0], limb, "V3F", "to", sides[1], limb, "RG", "F"))
            ] = options.EdgeOptions(
                source=join_str((sides[0], limb, "V3F")),
                target=join_str((sides[1], limb, "RG", "F")),
                weight=0.03,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )

            edges[
                join_str((sides[0], limb, "V3E", "to", sides[1], limb, "RG", "E"))
            ] = options.EdgeOptions(
                source=join_str((sides[0], limb, "V3E")),
                target=join_str((sides[1], limb, "RG", "E")),
                weight=0.02,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )

            edges[
                join_str((sides[0], limb, "V3E", "to", sides[1], limb, "RG", "In", "E"))
            ] = options.EdgeOptions(
                source=join_str((sides[0], limb, "V3E")),
                target=join_str((sides[1], limb, "RG", "In", "E")),
                weight=0.0,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )

            edges[
                join_str(
                    (sides[0], limb, "V3E", "to", sides[1], limb, "RG", "In", "E2")
                )
            ] = options.EdgeOptions(
                source=join_str((sides[0], limb, "V3E")),
                target=join_str((sides[1], limb, "RG", "In", "E2")),
                weight=0.8,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
    return edges


def connect_fore_hind_circuits():
    """Connect CPG's to Interneurons."""

    edges = {}
    for side in ("left", "right"):
        edges[join_str((side, "fore", "RG", "F", "to", side, "fore", "Ini", "hom"))] = (
            options.EdgeOptions(
                source=join_str((side, "fore", "RG", "F")),
                target=join_str((side, "fore", "Ini", "hom")),
                weight=0.70,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
        )

        edges[join_str((side, "fore", "RG", "F", "to", side, "V0D", "diag"))] = (
            options.EdgeOptions(
                source=join_str((side, "fore", "RG", "F")),
                target=join_str((side, "V0D", "diag")),
                weight=0.50,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
        )

        edges[join_str((side, "fore", "Ini", "hom", "to", side, "hind", "RG", "F"))] = (
            options.EdgeOptions(
                source=join_str((side, "fore", "Ini", "hom")),
                target=join_str((side, "hind", "RG", "F")),
                weight=-0.01,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
        )

        edges[join_str((side, "fore", "Sh2", "hom", "to", side, "hind", "RG", "F"))] = (
            options.EdgeOptions(
                source=join_str((side, "fore", "Sh2", "hom")),
                target=join_str((side, "hind", "RG", "F")),
                weight=0.01,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
        )

        edges[join_str((side, "hind", "Sh2", "hom", "to", side, "fore", "RG", "F"))] = (
            options.EdgeOptions(
                source=join_str((side, "hind", "Sh2", "hom")),
                target=join_str((side, "fore", "RG", "F")),
                weight=0.05,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
        )

        edges[
            join_str((side, "fore", "RG", "F", "to", side, "fore", "V0V", "diag"))
        ] = options.EdgeOptions(
            source=join_str((side, "fore", "RG", "F")),
            target=join_str((side, "fore", "V0V", "diag")),
            weight=0.325,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )

        edges[join_str((side, "hind", "RG", "F", "to", side, "hind", "V3", "diag"))] = (
            options.EdgeOptions(
                source=join_str((side, "hind", "RG", "F")),
                target=join_str((side, "hind", "V3", "diag")),
                weight=0.325,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
        )

        # edges[join_str((side, "fore", "V2a-diag", "to", side, "fore", "V0V", "diag"))] = options.EdgeOptions(
        #     source=join_str((side, "fore", "V2a-diag")),
        #     target=join_str((side, "fore", "V0V", "diag")),
        #     weight=0.9,
        #     type="excitatory",
        #     visual=options.EdgeVisualOptions(),
        # )

        # edges[join_str((side, "hind", "V2a-diag", "to", side, "hind", "V3", "diag"))] = options.EdgeOptions(
        #     source=join_str((side, "hind", "V2a-diag")),
        #     target=join_str((side, "hind", "V3", "diag")),
        #     weight=0.9,
        #     type="excitatory",
        #     visual=options.EdgeVisualOptions(),
        # )

    for sides in (("left", "right"), ("right", "left")):
        edges[
            join_str((sides[0], "V0D", "diag", "to", sides[1], "hind", "RG", "F"))
        ] = options.EdgeOptions(
            source=join_str((sides[0], "V0D", "diag")),
            target=join_str((sides[1], "hind", "RG", "F")),
            weight=-0.01,
            type="inhibitory",
            visual=options.EdgeVisualOptions(),
        )

        edges[
            join_str(
                (sides[0], "fore", "V0V", "diag", "to", sides[1], "hind", "RG", "F")
            )
        ] = options.EdgeOptions(
            source=join_str((sides[0], "fore", "V0V", "diag")),
            target=join_str((sides[1], "hind", "RG", "F")),
            weight=0.005,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )

        edges[
            join_str(
                (sides[0], "hind", "V3", "diag", "to", sides[1], "fore", "RG", "F")
            )
        ] = options.EdgeOptions(
            source=join_str((sides[0], "hind", "V3", "diag")),
            target=join_str((sides[1], "fore", "RG", "F")),
            weight=0.04,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )

    return edges


def generate_network():
    """Generate network"""

    N_ITERATIONS = int(1e4)

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "mouse"},
        logs=options.NetworkLogOptions(
            n_iterations=N_ITERATIONS,
        )
    )

    # Generate rhythm centers
    scale = 1.0
    for side in ("left", "right"):
        for limb in ("fore", "hind"):
            # Rhythm
            rg_x, rg_y = 10.0, 7.5
            off_x = -rg_x if side == "left" else rg_x
            off_y = rg_y if limb == "fore" else -rg_y
            mirror_x = limb == "hind"
            mirror_y = side == "right"
            rhythm = RhythmGenerator(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((rhythm.nodes()).values())
            network_options.add_edges((rhythm.edges()).values())
            # Rhtyhm Drive
            rhythm_drive = RhythmDrive(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((rhythm_drive.nodes()).values())
            # Pattern
            pf_x, pf_y = rg_x + 0.0, rg_y + 7.5
            off_x = -pf_x if side == "left" else pf_x
            off_y = pf_y if limb == "fore" else -pf_y
            mirror_x = limb == "hind"
            mirror_y = side == "right"
            pattern = PatternFormation(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((pattern.nodes()).values())
            network_options.add_edges((pattern.edges()).values())

            # Commissural
            comm_x, comm_y = rg_x - 7.0, rg_y + 0.0
            off_x = -comm_x if side == "left" else comm_x
            off_y = comm_y if limb == "fore" else -comm_y
            mirror_x = limb == "hind"
            mirror_y = side == "right"
            commissural = Commissural(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((commissural.nodes()).values())
            # Drive
            commissural_drive = CommissuralDrive(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((commissural_drive.nodes()).values())
        # LPSN
        lpsn_x = rg_x - 9.0
        lpsn_y = rg_y - 5.5
        off_x = -lpsn_x if side == "left" else lpsn_x
        off_y = lpsn_y
        mirror_y = side == "right"
        lpsn = LPSN(
            name=side,
            transform_mat=get_transform_mat(
                angle=0,
                off_x=off_x,
                off_y=off_y,
                mirror_y=mirror_y,
            ),
        )
        network_options.add_nodes((lpsn.nodes()).values())
        lpsn_drive = LPSNDrive(
                name=join_str((side,)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
        network_options.add_nodes((lpsn_drive.nodes()).values())

    #################################
    # Connect rhythm to commissural #
    #################################
    rg_commissural_edges = connect_rg_commissural()
    network_options.add_edges(rg_commissural_edges.values())

    ##############################
    # Connect fore and hind lpsn #
    ##############################
    fore_hind_edges = connect_fore_hind_circuits()
    network_options.add_edges(fore_hind_edges.values())

    for side in ("left", "right"):
        for limb in ("fore", "hind"):
            network_options.add_edge(
                options.EdgeOptions(
                    source=join_str((side, limb, "RG", "F", "DR")),
                    target=join_str((side, limb, "RG", "F")),
                    weight=1.0,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )
            network_options.add_edge(
                options.EdgeOptions(
                    source=join_str((side, limb, "RG", "E", "DR")),
                    target=join_str((side, limb, "RG", "E")),
                    weight=1.0,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )
            network_options.add_edge(
                options.EdgeOptions(
                    source=join_str((side, limb, "V0V", "DR")),
                    target=join_str((side, limb, "V0V")),
                    weight=-1.0,
                    type="inhibitory",
                    visual=options.EdgeVisualOptions(),
                )
            )
            network_options.add_edge(
                options.EdgeOptions(
                    source=join_str((side, limb, "V0D", "DR")),
                    target=join_str((side, limb, "V0D")),
                    weight=-1.0,
                    type="inhibitory",
                    visual=options.EdgeVisualOptions(),
                )
            )
        network_options.add_edge(
            options.EdgeOptions(
                source=join_str((side, "V0D", "diag", "DR")),
                target=join_str((side, "V0D", "diag")),
                weight=-1.0,
                type="inhibitory",
                visual=options.EdgeVisualOptions(),
            )
        )

    data = NetworkData.from_options(network_options)

    network = PyNetwork.from_options(network_options)

    # nnodes = len(network_options.nodes)
    # integrator.set_initial_value(np.zeros(len(data.states.array),), 0.0)

    # print("Data ------------", np.array(network.data.states.array))

    # data.to_file("/tmp/sim.hdf5")

    # integrator.integrate(integrator.t + 1e-3)

    # # Integrate
    states = np.ones((len(data.states.array),)) * 1.0

    # network_gui = NetworkGUI(data=data)
    # network_gui.run()

    # for index, node in enumerate(network_options.nodes):
    #     print(index, node.name)
    for iteration in tqdm(range(0, N_ITERATIONS), colour="green", ascii=" >="):
        network.data.external_inputs.array[:] = (
            np.ones((1,)) * (iteration / N_ITERATIONS) * 1.0
        )
        states = rk4(iteration * 1e-3, states, network.ode, step_size=1)
        network.logging(iteration)

    network.data.to_file("/tmp/network.h5")

    plt.figure()
    plt.fill_between(
        np.linspace(0.0, N_ITERATIONS * 1e-3, N_ITERATIONS),
        np.array(network.data.nodes[15].output.array),
        alpha=0.2,
        lw=1.0,
    )
    plt.plot(
        np.linspace(0.0, N_ITERATIONS * 1e-3, N_ITERATIONS),
        np.array(network.data.nodes[15].output.array),
        label="RG-F"
    )
    plt.fill_between(
        np.linspace(0.0, N_ITERATIONS * 1e-3, N_ITERATIONS),
        np.array(network.data.nodes[1].output.array),
        alpha=0.2,
        lw=1.0,
    )
    plt.plot(
        np.linspace(0.0, N_ITERATIONS * 1e-3, N_ITERATIONS),
        np.array(network.data.nodes[1].output.array),
        label="RG-E"
    )
    plt.legend()

    network_options.save("/tmp/netwok_options.yaml")

    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="source",
        target="target",
    )

    plt.figure()
    pos_circular = nx.circular_layout(graph)
    pos_spring = nx.spring_layout(graph)
    pos_graphviz = nx.nx_agraph.pygraphviz_layout(graph)

    _ = nx.draw_networkx_nodes(
        graph,
        pos={
            node: data["visual"]["position"][:2] for node, data in graph.nodes.items()
        },
        node_color=[data["visual"]["color"] for node, data in graph.nodes.items()],
        alpha=0.25,
        edgecolors="k",
        linewidths=2.0,
    )
    nx.draw_networkx_labels(
        graph,
        pos={
            node: data["visual"]["position"][:2] for node, data in graph.nodes.items()
        },
        labels={node: data["visual"]["label"] for node, data in graph.nodes.items()},
        font_size=11.0,
        font_weight="bold",
        font_family="sans-serif",
        alpha=1.0,
    )
    nx.draw_networkx_edges(
        graph,
        pos={
            node: data["visual"]["position"][:2] for node, data in graph.nodes.items()
        },
        edge_color=[
            [0.3, 1.0, 0.3]
            if data["type"] == "excitatory"
            else [0.7, 0.3, 0.3]
            for edge, data in graph.edges.items()
        ],
        width=1.0,
        arrowsize=10,
        style="dashed",
        arrows=True,
        min_source_margin=5,
        min_target_margin=5,
        connectionstyle=[
            data["visual"]["connectionstyle"]
            for edge, data in graph.edges.items()
        ],
    )
    plt.show()

    # generate_tikz_figure(
    #     graph,
    #     paths.get_project_data_path().joinpath("templates", "network",),
    #     "tikz-full-network.tex",
    #     paths.get_project_images_path().joinpath("quadruped_network.tex")
    # )


def generate_tikz_figure(
    network, template_env, template_name, export_path, add_axis=False, add_label=False
):
    """Generate tikz network"""

    ##########################################
    # Remove isolated nodes from the network #
    ##########################################
    network.remove_nodes_from(list(nx.isolates(network)))

    node_options = {
        name: node.get("neuron_class", "interneuron")
        for name, node in network.nodes.items()
    }

    options = {
        "flexor": "flexor-edge",
        "extensor": "extensor-edge",
        "excitatory": "excitatory-edge",
        "inhibitory": "inhibitory-edge",
        "interneuron": "inhibitory-edge",
    }
    edge_options = {
        edge: "{}, opacity={}".format(
            options.get(
                network.nodes[edge[0]].get("neuron_class", "interneuron"),
                "undefined-edge",
            ),
            # max(min(abs(data["weight"]), 1.0), 0.5)
            1.0,
        )
        for edge, data in network.edges.items()
    }

    raw_latex = nx.to_latex_raw(
        network,
        pos={name: (node["x"], node["y"]) for name, node in network.nodes.items()},
        node_options=node_options,
        # default_node_options="my-node",
        node_label={name: node["label"] for name, node in network.nodes.items()},
        # edge_label={
        #     name: np.round(edge['weight'], decimals=2)
        #     for name, edge in network.edges.items()
        # },
        edge_label_options={
            name: "fill=white, font={\\tiny}, opacity=1.0"
            for name, edge in network.edges.items()
        },
        default_edge_options=(
            "[color=black, ultra thick, -{Latex[scale=1.0]}, on background layer, opacity=1.0,]"  # auto=mid
        ),
        edge_options=edge_options,
    )

    # Render the network
    rhythm_groups = defaultdict(list)
    pattern_groups = defaultdict(list)
    commissural_groups = defaultdict(list)
    lpsn_groups = defaultdict(list)
    muscle_sensors_groups = defaultdict(list)
    for name, node in network.nodes.items():
        if node["neuron_class"] == "sensory":
            muscle_sensors_groups[node["neuron_group"]].append(name)
        if node.get("neuron_group") == "rhythm":
            rhythm_groups[node["layer"]].append(name)
        if node.get("neuron_group") == "pattern":
            pattern_groups[node["layer"]].append(name)
        if node.get("neuron_group") == "commissural":
            commissural_groups[node["layer"]].append(name)
        if node.get("neuron_group") == "LPSN":
            lpsn_groups[node["layer"]].append(name)

    environment = Environment(loader=FileSystemLoader(template_env))
    template = environment.get_template(template_name)
    content = template.render(
        network="\n".join(raw_latex.split("\n")[2:-2]),
        rhythm_groups=list(rhythm_groups.values()),
        pattern_groups=list(pattern_groups.values()),
        commissural_groups=list(commissural_groups.values()),
        lpsn_groups=list(lpsn_groups.values()),
        muscle_sensors_groups=list(muscle_sensors_groups.values()),
        add_axis=add_axis,
        add_legend=add_label,
    )
    with open(export_path, mode="w", encoding="utf-8") as message:
        message.write(content)

    result = os.system(
        f"pdflatex --shell-escape -output-directory={str(Path(export_path).parents[0])} {export_path}"
    )


def main():
    """Main."""

    # Generate the network
    generate_network()

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
