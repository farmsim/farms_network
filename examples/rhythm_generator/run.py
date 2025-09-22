import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from typing import Iterable, List
from farms_core.io.yaml import read_yaml
from farms_core.utils import profile
from farms_network.core import options
from farms_network.core.data import NetworkData
from farms_network.core.network import Network
from farms_network.numeric.integrators_cy import RK4Solver
from scipy.integrate import ode
from tqdm import tqdm

plt.rcParams['text.usetex'] = False


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
        parameters = options.LINaPDannerNodeParameterOptions.defaults(**parameters)
        noise = options.OrnsteinUhlenbeckOptions.defaults()
        node_options_class = options.LINaPDannerNodeOptions
    elif node_type == "LIDanner":
        state_options = options.LIDannerStateOptions.from_kwargs(**states)
        parameters = options.LIDannerNodeParameterOptions.defaults(**parameters)
        noise = options.OrnsteinUhlenbeckOptions.defaults()
        node_options_class = options.LIDannerNodeOptions
    elif node_type == "Linear":
        state_options = None
        parameters = options.LinearParameterOptions.defaults(**parameters)
        noise = None
        node_options_class = options.LinearNodeOptions
    elif node_type == "ReLU":
        state_options = None
        parameters = options.ReLUParameterOptions.defaults(**parameters)
        noise = None
        node_options_class = options.ReLUNodeOptions
    elif node_type == "ExternalRelay":
        state_options = None
        parameters = options.NodeParameterOptions()
        noise = None
        visual_options.radius = 0.0
        node_options_class = options.RelayNodeOptions
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


class BrainStemDrive:
    """ Generate Brainstem drive network """

    def __init__(self, name="", transform_mat=np.identity(3)):
        """Initialization."""
        super().__init__()
        self.name = name
        self.transform_mat = transform_mat

    def nodes(self):
        """Add nodes."""
        node_specs = [
            (
                join_str(("BS", "input")),
                "Relay",
                np.array((3.0, 0.0)),
                "A",
                [0.0, 0.0, 0.0],
                {},
                {},
            ),
            (
                join_str(("BS", "DR")),
                "Linear",
                np.array((3.0, -1.0)),
                "A",
                [0.0, 0.0, 0.0],
                None,
                {"slope": 1.0, "bias": 0.0},
            ),
        ]

        # Loop through the node specs to create each node using the create_node function
        nodes = create_nodes(node_specs, self.name, self.transform_mat)
        return nodes

    def edges(self):
        """Add edges."""

        # Define edge details in a list for easier iteration
        edge_specs = [
            (("BS", "input"), ("BS", "DR"), 1.0, "excitatory"),
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
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
                {"v": -60.0, "a": 0.0},
                {},
            ),
            (
                join_str(("RG", "In", "E")),
                "LIDanner",
                np.array((-1.0, 1.5)),
                "In",
                [0.2, 0.2, 0.2],
                {"v": -60.0, "a": 0.0},
                {},
            ),
            # (
            #     join_str(("RG", "In", "E2")),
            #     "LIDanner",
            #     np.array((-5.0, 1.0)),
            #     "In",
            #     [0.2, 0.2, 0.2],
            #     {"v": -60.0, "a": 0.0},
            #     {},
            # ),
            # (
            #     join_str(("RG", "F", "DR")),
            #     "Linear",
            #     np.array((3.0, 2.0)),
            #     "d",
            #     [0.5, 0.5, 0.5],  # Default visual color if needed
            #     None,
            #     {"slope": 0.1, "bias": 0.0},
            # ),
            # (
            #     join_str(("RG", "E", "DR")),
            #     "Linear",
            #     np.array((-3.0, 2.0)),
            #     "d",
            #     [0.5, 0.5, 0.5],  # Default visual color if needed
            #     None,
            #     {"slope": 0.0, "bias": 0.1},
            # ),
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
            # (("RG", "In", "E2"), ("RG", "F"), -0.04, "inhibitory"),
            # (("RG", "F", "DR"), ("RG", "F"), 1.0, "excitatory"),
            # (("RG", "E", "DR"), ("RG", "E"), 1.0, "excitatory"),
        ]

        # Loop through the edge specs to create each edge
        edges = create_edges(edge_specs, self.name)
        return edges


# class RhythmGenerator:
#     """Generate RhythmGenerator Network"""

#     def __init__(self, name="", anchor_x=0.0, anchor_y=0.0):
#         """Initialization."""
#         super().__init__()
#         self.name = name

#     def nodes(self):
#         """Add nodes."""
#         nodes = {}
#         nodes["RG-F"] = options.LINaPDannerNodeOptions(
#             name=self.name + "-RG-F",
#             parameters=options.LINaPDannerNodeParameterOptions.defaults(),
#             visual=options.NodeVisualOptions(label="F", color=[1.0, 0.0, 0.0]),
#             state=options.LINaPDannerStateOptions.from_kwargs(
#                 v=-60.5, h=np.random.uniform(0, 1)
#             ),
#             noise=None
#         )

#         nodes["RG-E"] = options.LINaPDannerNodeOptions(
#             name=self.name + "-RG-E",
#             parameters=options.LINaPDannerNodeParameterOptions.defaults(),
#             visual=options.NodeVisualOptions(label="E", color=[0.0, 1.0, 0.0]),
#             state=options.LINaPDannerStateOptions.from_kwargs(
#                 v=-62.5, h=np.random.uniform(0, 1)
#             ),
#             noise=None
#         )

#         nodes["In-F"] = options.LIDannerNodeOptions(
#             name=self.name + "-In-F",
#             parameters=options.LIDannerNodeParameterOptions.defaults(),
#             visual=options.NodeVisualOptions(label="In", color=[0.2, 0.2, 0.2]),
#             state=options.LIDannerStateOptions.from_kwargs(v=-60.0, a=0.0),
#             noise=None
#         )

#         nodes["In-E"] = options.LIDannerNodeOptions(
#             name=self.name + "-In-E",
#             parameters=options.LIDannerNodeParameterOptions.defaults(),
#             visual=options.NodeVisualOptions(label="In", color=[0.2, 0.2, 0.2]),
#             state=options.LIDannerStateOptions.from_kwargs(v=-60.0, a=0.0),
#             noise=None
#         )
#         return nodes

#     def edges(self):
#         edges = {}
#         edges["RG-F-to-In-F"] = options.EdgeOptions(
#             source=self.name + "-RG-F",
#             target=self.name + "-In-F",
#             weight=0.4,
#             type="excitatory",
#             visual=options.EdgeVisualOptions(),
#         )
#         edges["In-F-to-RG-E"] = options.EdgeOptions(
#             source=self.name + "-In-F",
#             target=self.name + "-RG-E",
#             weight=-1.0,
#             type="inhibitory",
#             visual=options.EdgeVisualOptions(),
#         )
#         edges["In-E-to-RG-F"] = options.EdgeOptions(
#             source=self.name + "-In-E",
#             target=self.name + "-RG-F",
#             weight=-0.08,
#             type="inhibitory",
#             visual=options.EdgeVisualOptions(),
#         )
#         edges["RG-E-to-In-E"] = options.EdgeOptions(
#             source=self.name + "-RG-E",
#             target=self.name + "-In-E",
#             weight=0.4,
#             type="excitatory",
#             visual=options.EdgeVisualOptions(),
#         )
#         return edges


def generate_network(iterations=1000):
    """ Generate network """

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "rhythm_generator"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=iterations,
            timestep=float(1.0),
        ),
        logs=options.NetworkLogOptions(
            n_iterations=iterations,
            buffer_size=iterations,
        )
    )

    # Generate rhythm center
    rhythm = RhythmGenerator(name="")
    network_options.add_nodes((rhythm.nodes()).values())
    network_options.add_edges((rhythm.edges()).values())

    flexor_drive = options.LinearNodeOptions(
        name="FD",
        parameters=options.LinearParameterOptions.defaults(slope=0.1, bias=0.0),
        visual=options.NodeVisualOptions(position=(1.0, 0.0)),
        noise=None
    )
    extensor_drive = options.LinearNodeOptions(
        name="ED",
        parameters=options.LinearParameterOptions.defaults(slope=0.0, bias=0.1),
        visual=options.NodeVisualOptions(position=(1.0, 1.0)),
        noise=None
    )

    drive = options.RelayNodeOptions(
        name="D",
        visual=options.NodeVisualOptions(position=(5.0, 5.0)),
        parameters=None,
        noise=None
    )

    network_options.add_node(flexor_drive)
    network_options.add_node(extensor_drive)
    network_options.add_node(drive)

    network_options.add_edge(
        options.EdgeOptions(
            source="FD",
            target="RG_F",
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
    )
    network_options.add_edge(
        options.EdgeOptions(
            source="ED",
            target="RG_E",
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
    )
    network_options.add_edge(
        options.EdgeOptions(
            source="ED",
            target="RG_In_E",
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
    )
    network_options.add_edge(
        options.EdgeOptions(
            source="D",
            target="ED",
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
    )
    network_options.add_edge(
        options.EdgeOptions(
            source="D",
            target="FD",
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
    )


    # network_options = options.NetworkOptions.from_options(
    #     read_yaml("/Users/tatarama/projects/work/farms/farms_network/examples/mouse/config/network_options.yaml")
    # )
    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="source",
        target="target"
    )
    # node_positions = nx.spring_layout(graph)
    # network_options.save("/tmp/rhythm.yaml")
    # for index, node in enumerate(network_options.nodes):
    #     node.visual.position[:2] = node_positions[node.name]

    data = NetworkData.from_options(network_options)

    network = Network.from_options(network_options)

    # network.setup_integrator(network_options)
    rk4solver = RK4Solver(network.nstates, network_options.integration.timestep)

    integrator = ode(network.get_ode_func()).set_integrator(
        u'dopri5',
        method=u'adams',
        max_step=0.0,
        # nsteps=0
    )
    nnodes = len(network_options.nodes)
    integrator.set_initial_value(np.zeros(len(data.states.array[:]),), 0.0)

    # print("Data ------------", np.array(network.data.states.array))

    # data.to_file("/tmp/sim.hdf5")

    # # Integrate
    states = np.ones((iterations, len(data.states.array[:])))*1.0
    states_tmp = np.zeros((len(data.states.array[:],)))
    outputs = np.ones((iterations, len(data.outputs.array[:])))*1.0
    # states[0, 2] = -1.0

    # for index, node in enumerate(network_options.nodes):
    #     print(index, node.name)
    # network.data.external_inputs.array[:] = np.ones((1,))*(iteration/iterations)*1.0
    inputs = np.ones(np.shape(network.data.external_inputs.array[:]))
    # print(np.array(network.data.connectivity.weights), np.array(network.data.connectivity.edge_indices), np.array(network.data.connectivity.node_indices), np.array(network.data.connectivity.index_offsets))
    for iteration in tqdm(range(0, iterations), colour='green', ascii=' >='):
        time = iteration
        # network.step(network.ode, iteration*1e-3, network.data.states.array)
        # network.step()
        # states[iteration+1, :] = network.data.states.array
        # network.step()
        # network.evaluate(iteration*1e-3, states[iteration, :])

        _iter = network._network_cy.iteration
        network.data.times.array[_iter] = time
        network.data.external_inputs.array[:] = inputs*0.5
        # integrator.set_initial_value(integrator.y, integrator.t)
        # integrator.integrate(integrator.t+1.0)
        # network.data.states.array[:] = integrator.y
        rk4solver.step(network._network_cy, time, network.data.states.array)
        # outputs[iteration, :] = network.data.outputs.array
        # states[iteration, :] = integrator.y# network.data.states.array
        # network._network_cy.update_iteration()
        network._network_cy.iteration += 1
        network._network_cy.update_logs(network._network_cy.iteration)

    # network.data.to_file("/tmp/network.h5")
    nodes_data = network.log.nodes
    plt.figure()
    plt.plot(
        np.linspace(0.0, iterations*1e-3, iterations), states[:, :],
    )
    plt.figure()
    plt.fill_between(
        np.linspace(0.0, iterations*1e-3, iterations), np.asarray(nodes_data[0].output.array),
        alpha=0.2, lw=1.0,
    )
    plt.plot(
        np.linspace(0.0, iterations*1e-3, iterations), np.asarray(nodes_data[0].output.array),
        label="RG-F"
    )
    plt.fill_between(
        np.linspace(0.0, iterations*1e-3, iterations), np.asarray(nodes_data[1].output.array),
        alpha=0.2, lw=1.0,
    )
    plt.plot(
        np.linspace(0.0, iterations*1e-3, iterations), np.asarray(nodes_data[1].output.array), label="RG-E"
    )
    plt.legend()

    plt.figure()
    # node_positions = nx.circular_layout(graph)
    # node_positions = nx.forceatlas2_layout(graph)
    node_positions = {}
    for index, node in enumerate(network_options.nodes):
        node_positions[node.name] = node.visual.position[:2]

    _ = nx.draw_networkx_nodes(
        graph,
        pos=node_positions,
        node_color=[data["visual"]["color"] for node, data in graph.nodes.items()],
        alpha=0.25,
        edgecolors='k',
        linewidths=2.0,
    )
    nx.draw_networkx_labels(
        graph,
        pos=node_positions,
        labels={node: data["visual"]["label"] for node, data in graph.nodes.items()},
        font_size=11.0,
        font_weight='bold',
        font_family='sans-serif',
        alpha=1.0,
    )
    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        edge_color=[
            [0.0, 1.0, 0.0] if data["type"] == "excitatory" else [1.0, 0.0, 0.0]
            for edge, data in graph.edges.items()
        ],
        width=1.,
        arrowsize=10,
        style='dashed',
        arrows=True,
        min_source_margin=5,
        min_target_margin=5,
        connectionstyle="arc3,rad=-0.2",
    )
    # plt.figure()
    # sparse_array = nx.to_scipy_sparse_array(graph)
    # sns.heatmap(
    #     sparse_array.todense(), cbar=False, square=True,
    #     linewidths=0.5,
    #     annot=True
    # )
    plt.show()

    # generate_tikz_figure(
    #     graph,
    #     paths.get_project_data_path().joinpath("templates", "network",),
    #     "tikz-full-network.tex",
    #     paths.get_project_images_path().joinpath("quadruped_network.tex")
    # )


def main():
    """Main."""

    # Generate the network
    # profile.profile(generate_network)
    generate_network()

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
