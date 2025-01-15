"""
Hopf Oscillator

[1]L. Righetti and A. J. Ijspeert, “Pattern generators with sensory
feedback for the control of quadruped locomotion,” in 2008 IEEE
International Conference on Robotics and Automation, May 2008,
pp. 819–824. doi: 10.1109/ROBOT.2008.4543306.
"""


import farms_pylog as pylog
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from farms_core.utils import profile
from farms_network.core import options
from farms_network.core.data import NetworkData
from farms_network.core.network import Network
from tqdm import tqdm

plt.rcParams['text.usetex'] = False


def join_strings(strings):
    return "_".join(strings)


class RhythmDrive:
    """ Generate Drive Network """

    def __init__(self, name="", anchor_x=0.0, anchor_y=0.0):
        """Initialization."""
        super().__init__()
        self.name = name

    def nodes(self):
        """Add nodes."""
        nodes = {}
        name = join_strings((self.name, "RG", "F", "DR"))
        nodes[name] = options.LinearNodeOptions(
            name=name,
            parameters=options.LinearParameterOptions.defaults(slope=0.1, bias=0.0),
            visual=options.NodeVisualOptions(),
        )
        name = join_strings((self.name, "RG", "E", "DR"))
        nodes[name] = options.LinearNodeOptions(
            name=name,
            parameters=options.LinearParameterOptions.defaults(slope=0.0, bias=0.1),
            visual=options.NodeVisualOptions(),
        )

        return nodes


def generate_network(iterations=20000):
    """ Generate network """

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "rhigetti08"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=iterations,
            timestep=float(1e-3),
        ),
        logs=options.NetworkLogOptions(
            n_iterations=iterations,
            buffer_size=iterations,
        )
    )

    # Generate rhythm centers
    n_oscillators = 4

    # Neuron
    # Create an oscillator for each joint
    num_oscillators = 4
    oscillator_names = [f'n{num}' for num in range(num_oscillators)]
    for j, neuron_name in enumerate(oscillator_names):
        network_options.add_node(
            options.HopfOscillatorNodeOptions(
                name=neuron_name,
                parameters=options.HopfOscillatorNodeParameterOptions.defaults(
                    mu=1.0,
                    omega=5.0,
                    alpha=5.0,
                    beta=5.0,
                ),
                visual=options.NodeVisualOptions(
                    label=f"{j}", color=[1.0, 0.0, 0.0]
                ),
                state=options.HopfOscillatorStateOptions.from_kwargs(
                    x=np.random.uniform(0.0, 1.0),
                    y=np.random.uniform(0.0, 1.0),
                ),
                noise=None,
            )
        )

    # Connect edges
    connection_matrix_walk = np.asarray(
        [
            [0, -1, 1, -1],
            [-1, 0, -1, 1],
            [-1, 1, 0, -1],
            [1, -1, -1, 0]
        ]
    ).T

    connection_matrix_trot = np.asarray(
        [
            [0, -1, -1, 1],
            [-1, 0, 1, -1],
            [-1, 1, 0, -1],
            [1, -1, -1, 0]
        ]
    ).T

    for i, j in zip(*np.nonzero(connection_matrix_trot)):
        network_options.add_edge(
            options.EdgeOptions(
                source=oscillator_names[i],
                target=oscillator_names[j],
                weight=connection_matrix_trot[i, j]*1,
                type="excitatory",
                visual=options.EdgeVisualOptions(),
            )
        )

    data = NetworkData.from_options(network_options)

    network = Network.from_options(network_options)
    network.setup_integrator(network_options)

    # nnodes = len(network_options.nodes)
    # integrator.set_initial_value(np.zeros(len(data.states.array),), 0.0)

    # print("Data ------------", np.array(network.data.states.array))

    # data.to_file("/tmp/sim.hdf5")

    # integrator.integrate(integrator.t + 1e-3)

    # # Integrate
    states = np.ones((iterations+1, len(data.states.array)))*0.0
    outputs = np.ones((iterations, len(data.outputs.array)))*1.0
    # states[0, 2] = -1.0

    # for index, node in enumerate(network_options.nodes):
    #     print(index, node.name)
    # network.data.external_inputs.array[:] = np.ones((1,))*(iteration/iterations)*1.0
    for iteration in tqdm(range(0, iterations), colour='green', ascii=' >='):
        # network.step(network.ode, iteration*1e-3, network.data.states.array)
        # network.step()
        # states[iteration+1, :] = network.data.states.array
        network.step()
        network.data.times.array[iteration] = iteration*1e-3

    # network.data.to_file("/tmp/network.h5")
    plt.figure()
    for j in range(n_oscillators):
        plt.fill_between(
            np.array(network.data.times.array),
            2*j + (1 + np.sin(network.data.nodes[j].output.array)),
            2*j,
            alpha=0.2,
            lw=1.0,
        )
        plt.plot(
            np.array(network.data.times.array),
            2*j + (1 + np.sin(network.data.nodes[j].output.array)),
            label=f"{j}"
        )
    plt.legend()

    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="source",
        target="target"
    )
    plt.figure()
    node_positions = nx.circular_layout(graph)
    node_positions = nx.spring_layout(graph)
    for index, node in enumerate(network_options.nodes):
        node.visual.position[:2] = node_positions[node.name]

    network_options.save("/tmp/network_options.yaml")

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
    plt.figure()
    sparse_array = nx.to_scipy_sparse_array(graph)
    sns.heatmap(
        sparse_array.todense(), cbar=False, square=True,
        linewidths=0.5,
        annot=True
    )
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
    profile.profile(generate_network)

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
