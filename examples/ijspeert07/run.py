""" Generate and reproduce Ijspeert 07 Science paper
DOI: 10.1126/science.1138353 """


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from farms_core.utils import profile
from farms_network.core import options
from farms_core.io.yaml import read_yaml
from farms_network.core.data import NetworkData
from farms_network.core.network import Network
from tqdm import tqdm
from farms_network.numeric.integrators_cy import RK4Solver
from scipy.integrate import ode, RK45, RK23

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
    intrinsic_frequency = kwargs.get('intrinsic_frequency', 1.0)
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
                state=options.OscillatorStateOptions(
                    initial=[
                        np.random.uniform(-np.pi, np.pi),
                        np.random.uniform(0, 1),
                        np.random.uniform(0, 1)
                    ]
                ),
                noise=None,
            )
        )
    # Connect
    phase_diff = kwargs.get('axial_phi', -np.pi/2)
    weight = kwargs.get('axial_w', 1e4)
    connections = np.vstack(
        (np.arange(n_oscillators),
         np.roll(np.arange(n_oscillators), -1)))[:, :-1]
    for j in np.arange(n_oscillators-1):
        network_options.add_edge(
            options.OscillatorEdgeOptions(
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
            options.OscillatorEdgeOptions(
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
    weight = kwargs.get('anti_w', 1e4)
    for n in range(n_oscillators):
        network_options.add_edge(
            options.OscillatorEdgeOptions(
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
            options.OscillatorEdgeOptions(
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


def generate_network(iterations=10000):
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
    n_oscillators = 10
    network_options = oscillator_double_chain(network_options, n_oscillators)
    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="source",
        target="target"
    )
    node_positions = nx.spring_layout(graph)
    for index, node in enumerate(network_options.nodes):
        node.visual.position[:2] = node_positions[node.name]
    return network_options


def run_network(network_options: options.NetworkOptions):
    """ Run network """

    network = Network.from_options(network_options)
    iterations = network_options.integration.n_iterations
    timestep = network_options.integration.timestep

    # Setup integrators
    rk4solver = RK4Solver(network.nstates, timestep)

    sc_integrator = RK45(
        network.get_ode_func(),
        t0=0.0,
        y0=np.zeros(network.nstates,),
        t_bound=iterations*timestep,
        # max_step=timestep,
        # first_step=timestep,
        # rtol=1e-2,
        # atol=1e2,
    )

    integrator = ode(network.get_ode_func()).set_integrator(
        'dopri5',
        max_step=timestep,          # your RK4 step
        nsteps=10000,
    )

    nnodes = len(network_options.nodes)
    integrator.set_initial_value(np.zeros(network.nstates,), 0.0)

    # Integrate
    states = np.ones((iterations+1, network.nstates))*1.0
    outputs = np.ones((iterations, network.nnodes))*1.0
    for iteration in tqdm(range(0, iterations), colour='green', ascii=' >='):
        network.data.times.array[iteration] = iteration*timestep

        integrator.set_initial_value(integrator.y, integrator.t)
        integrator.integrate(integrator.t+timestep)

        # sc_integrator.step()

        # rk4solver.step(network._network_cy, iteration*timestep, network.data.states.array)

        network._network_cy.update_logs(network._network_cy.iteration)
        network._network_cy.iteration += 1

    plt.figure()
    for j in range(int(network.nnodes/2)):
        plt.fill_between(
            np.array(network.log.times.array),
            2*j + (1 + np.sin(np.array(network.log.outputs.array[:, j]))),
            2*j,
            alpha=0.2,
            lw=1.0,
        )
        plt.plot(
            np.array(network.data.times.array),
            2*j + (1 + np.sin(network.log.outputs.array[:, j])),
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
    node_positions = nx.forceatlas2_layout(graph)
    for index, node in enumerate(network_options.nodes):
        node.visual.position[:2] = node_positions[node.name]

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


def main():
    """Main."""

    # Generate the network
    network = generate_network()
    profile.profile(run_network, network)

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
