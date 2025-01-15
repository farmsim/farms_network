"""Leaky integrator

[1] Beer RD. 1995. On the Dynamics of Small Continuous-Time Recurrent Neural Networks.
Adaptive Behavior 3:469–509. doi:10.1177/105971239500300405

"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from farms_core import pylog
from farms_core.utils import profile
from farms_network.core import options
from farms_network.core.data import NetworkData
from farms_network.core.network import PyNetwork
from tqdm import tqdm

plt.rcParams['text.usetex'] = False


def join_strings(strings):
    return "_".join(strings)


def generate_network(iterations=20000):
    """ Generate network """

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "beer95"},
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
    n_neurons = 2

    # Neuron
    # Create an neuron for each joint
    num_neurons = 2
    neuron_names = [f'n{num}' for num in range(num_neurons)]
    biases = [-2.75, -1.75]
    positions = [(0.0, -5.0), (0.0, 5.0)]
    for j, neuron_name in enumerate(neuron_names):
        network_options.add_node(
            options.LeakyIntegratorNodeOptions(
                name=neuron_name,
                parameters=options.LeakyIntegratorParameterOptions.defaults(
                    tau=0.1,
                    bias=biases[j],
                    D=1.0,
                ),
                visual=options.NodeVisualOptions(
                    label=f"{j}", color=[1.0, 0.0, 0.0]
                ),
                state=options.LeakyIntegratorStateOptions.from_kwargs(
                    m=np.random.uniform(0.0, 1.0),
                ),
                noise=None,
            )
        )

    # Connect edges
    connection_matrix = np.asarray(
        [
            [4.5, 1,],
            [-1, 4.5,],
        ]
    ).T

    for i, j in zip(*np.nonzero(connection_matrix)):
        weight = connection_matrix[i, j]
        print(f"{neuron_names[i]}-->{neuron_names[j]}={weight}")
        network_options.add_edge(
            options.EdgeOptions(
                source=neuron_names[i],
                target=neuron_names[j],
                weight=weight,
                type="excitatory" if weight > 0.0 else "inhibitory",
                visual=options.EdgeVisualOptions(),
            )
        )

    network_options.save("/tmp/beer95.yaml")

    network = PyNetwork.from_options(network_options)
    network.setup_integrator(network_options)
    data = network.data

    # # Integrate
    states = np.ones((iterations+1, len(data.states.array)))*0.0
    outputs = np.ones((iterations, len(data.outputs.array)))*1.0

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
    for j in range(n_neurons):
        plt.plot(
            np.array(network.data.times.array),
            np.asarray(network.data.nodes[j].output.array),
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
