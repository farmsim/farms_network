""" Generate and reproduce Zhang, Shevtsova, et al. eLife 2022;11:e73424. DOI:
https://doi.org/10.7554/eLife.73424 paper network """


import itertools
import os
from copy import deepcopy
from pprint import pprint

import farms_pylog as pylog
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from farms_core.io.yaml import read_yaml, write_yaml
from farms_core.utils import profile
from farms_network.core import options
from farms_network.core.data import (NetworkConnectivity, NetworkData,
                                     NetworkStates)
from farms_network.core.network import PyNetwork
from tqdm import tqdm
import seaborn as sns

plt.rcParams['text.usetex'] = False


def join_strings(strings):
    return "_".join(strings)


def oscillator_chain(network_options, n_oscillators, name_prefix, **kwargs):
    """ Create a chain of n-oscillators. """
    # Define a network graph

    oscillator_names = [
        "{}_{}".format(name_prefix, n)
        for n in range(n_oscillators)
    ]
    # Oscillators
    intrinsic_frequency = kwargs.get('intrinsic_frequency', 0.2)
    nominal_amplitude = kwargs.get('nominal_amplitude', 1.0)
    amplitude_rate = kwargs.get('amplitude_rate', 20.0)

    origin = kwargs.get('origin', [0, 0])
    for j, osc in enumerate(oscillator_names):
        network_options.add_node(
            options.OscillatorNodeOptions(
                name=osc,
                parameters=options.OscillatorNodeParameterOptions.defaults(
                    intrinsic_frequency=intrinsic_frequency,
                    nominal_amplitude=nominal_amplitude,
                    amplitude_rate=amplitude_rate,
                ),
                visual=options.NodeVisualOptions(
                    label=f"{j}", color=[1.0, 0.0, 0.0]
                ),
                state=options.OscillatorStateOptions.from_kwargs(
                    phase=np.random.uniform(-np.pi, np.pi),
                    amplitude=np.random.uniform(0, 1),
                    amplitude_0=np.random.uniform(0, 1)
                ),
            )
        )
    # Connect
    phase_diff = kwargs.get('axial_phi', 2*np.pi/8)
    weight = kwargs.get('axial_w', 5)
    connections = np.vstack(
        (np.arange(n_oscillators),
         np.roll(np.arange(n_oscillators), -1)))[:, :-1]
    for j in np.arange(n_oscillators-1):
        network_options.add_edge(
            options.EdgeOptions(
                source=oscillator_names[connections[0, j]],
                target=oscillator_names[connections[1, j]],
                weight=weight,
                type="excitatory",
                parameters=options.OscillatorEdgeParameterOptions(
                    phase_difference=-1*phase_diff
                ),
                visual=options.EdgeVisualOptions(),
            )
        )

        network_options.add_edge(
            options.EdgeOptions(
                source=oscillator_names[connections[1, j]],
                target=oscillator_names[connections[0, j]],
                weight=weight,
                type="excitatory",
                parameters=options.OscillatorEdgeParameterOptions(
                    phase_difference=phase_diff
                ),
                visual=options.EdgeVisualOptions(),
            )
        )
    return network_options


def oscillator_double_chain(network_options, n_oscillators, **kwargs):
    """ Create a double chain of n-oscillators. """
    kwargs['origin'] = [-0.05, 0]
    network_options = oscillator_chain(network_options, n_oscillators, 'left', **kwargs)
    kwargs['origin'] = [0.05, 0]
    network_options = oscillator_chain(network_options, n_oscillators, 'right', **kwargs)

    # Connect double chain
    phase_diff = kwargs.get('anti_phi', np.pi)
    weight = kwargs.get('anti_w', 50)
    for n in range(n_oscillators):
        network_options.add_edge(
            options.EdgeOptions(
                source=f'left_{n}',
                target=f'right_{n}',
                weight=weight,
                type="excitatory",
                parameters=options.OscillatorEdgeParameterOptions(
                    phase_difference=phase_diff
                ),
                visual=options.EdgeVisualOptions(),
            )
        )
        network_options.add_edge(
            options.EdgeOptions(
                source=f'right_{n}',
                target=f'left_{n}',
                weight=weight,
                type="excitatory",
                parameters=options.OscillatorEdgeParameterOptions(
                    phase_difference=phase_diff
                ),
                visual=options.EdgeVisualOptions(),
            )
        )
    return network_options


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


def generate_network(iterations=2000):
    """ Generate network """

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "ijspeert07"},
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
    n_oscillators = 6
    network_options = oscillator_double_chain(network_options, n_oscillators)

    data = NetworkData.from_options(network_options)

    network = PyNetwork.from_options(network_options)
    network.setup_integrator(network_options.integration)

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
    for j in range(int(n_oscillators/2)+1):
        plt.fill_between(
            np.array(network.data.times.array),
            j + (1 + np.sin(network.data.nodes[j].output.array)),
            j,
            alpha=0.2,
            lw=1.0,
        )
        plt.plot(
            np.array(network.data.times.array),
            j + (1 + np.sin(network.data.nodes[j].output.array)),
            label=f"{j}"
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
        target="target"
    )
    plt.figure()
    pos_circular = nx.circular_layout(graph)
    pos_spring = nx.spring_layout(graph)
    pos_graphviz = nx.nx_agraph.pygraphviz_layout(graph)
    _ = nx.draw_networkx_nodes(
        graph,
        pos=pos_graphviz,
        node_color=[data["visual"]["color"] for node, data in graph.nodes.items()],
        alpha=0.25,
        edgecolors='k',
        linewidths=2.0,
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos_graphviz,
        labels={node: data["visual"]["label"] for node, data in graph.nodes.items()},
        font_size=11.0,
        font_weight='bold',
        font_family='sans-serif',
        alpha=1.0,
    )
    nx.draw_networkx_edges(
        graph,
        pos=pos_graphviz,
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
