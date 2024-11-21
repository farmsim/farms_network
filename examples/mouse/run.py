""" Generate and reproduce Zhang, Shevtsova, et al. eLife 2022;11:e73424. DOI:
https://doi.org/10.7554/eLife.73424 paper network """

import seaborn as sns
from farms_core.io.yaml import read_yaml
from farms_core.utils import profile
from farms_network.core import options

from components import *
from components import limb_circuit


def generate_network(n_iterations: int):
    """Generate network"""

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "mouse"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=n_iterations,
            timestep=1.0,
        ),
        logs=options.NetworkLogOptions(
            n_iterations=n_iterations,
        ),
    )

    ##############
    # MotorLayer #
    ##############
    # read muscle config file
    muscles_config = read_yaml(
        "/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/muscles/quadruped_siggraph.yaml"
    )

    def update_muscle_name(name: str) -> str:
        """Update muscle name format"""
        return name.replace("_", "-")

    muscles = {
        "left": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
        "right": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
    }

    for name, muscle in muscles_config["muscles"].items():
        side = muscle["side"]
        limb = muscle["limb"]
        function = muscle.get("function", "agonist")
        muscles[side][limb][function].append(
            {
                "name": join_str(name.split("_")[2:]),
                "type": muscle["type"],
                "abbrev": muscle["abbrev"],
            }
        )

    ###################################
    # Connect patterns and motorlayer #
    ###################################
    hind_muscle_patterns = {
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
    }

    fore_muscle_patterns = {
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

            rhythm_pattern_edges = connect_rhythm_pattern(base_name=join_str((side, limb)))
            network_options.add_edges(rhythm_pattern_edges.values())

            # Motor Layer
            motor_x = pf_x + 0.5 * max(
                len(muscles["left"][limb]["agonist"]),
                len(muscles["left"][limb]["antagonist"]),
            )
            motor_y = pf_y + 5.0

            # Determine the mirror_x and mirror_y flags based on side and limb
            mirror_x = True if limb == "hind" else False
            mirror_y = True if side == "right" else False

            # Create MotorLayer for each side and limb
            motor = MotorLayer(
                muscles=muscles[side][limb],
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0.0,
                    off_x=motor_x if side == "right" else -motor_x,
                    off_y=motor_y if limb == "fore" else -motor_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((motor.nodes()).values())
            network_options.add_edges((motor.edges()).values())
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
            name=side,
            transform_mat=get_transform_mat(
                angle=0,
                off_x=off_x,
                off_y=off_y,
                mirror_x=mirror_x,
                mirror_y=mirror_y,
            ),
        )
        network_options.add_nodes((lpsn_drive.nodes()).values())

        # Connect pattern layer to motor layer
        for muscle, patterns in hind_muscle_patterns.items():
            pattern_edges = connect_pattern_motor_layer(
                base_name=join_str((side, "hind")), muscle=muscle, patterns=patterns
            )
            network_options.add_edges(pattern_edges.values())
        for muscle, patterns in fore_muscle_patterns.items():
            pattern_edges = connect_pattern_motor_layer(
                base_name=join_str((side, "fore")), muscle=muscle, patterns=patterns
            )
            network_options.add_edges(pattern_edges.values())

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

    edge_specs = []

    for side in ("left", "right"):
        for limb in ("fore", "hind"):
            edge_specs.extend([
                ((side, limb, "RG", "F", "DR"), (side, limb, "RG", "F"), 1.0, "excitatory"),
                ((side, limb, "RG", "E", "DR"), (side, limb, "RG", "E"), 1.0, "excitatory"),
                ((side, limb, "V0V", "DR"), (side, limb, "V0V"), -1.0, "inhibitory"),
                ((side, limb, "V0D", "DR"), (side, limb, "V0D"), -1.0, "inhibitory"),
            ])

        # Add the diagonal V0D connection
        edge_specs.append(
            ((side, "V0D", "diag", "DR"), (side, "V0D", "diag"), -1.0, "inhibitory")
        )

    # Create the edges using create_edges
    edges = create_edges(
        edge_specs,
        base_name="",
        visual_options=options.EdgeVisualOptions()
    )
    network_options.add_edges(edges.values())

    return network_options


def generate_limb_circuit(n_iterations: int):
    """ Generate limb circuit """
    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "mouse"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=n_iterations,
            timestep=1.0,
        ),
        logs=options.NetworkLogOptions(
            n_iterations=n_iterations,
        ),
    )

     ##############
    # MotorLayer #
    ##############
    # read muscle config file
    muscles_config = read_yaml(
        "/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/muscles/quadruped_siggraph.yaml"
    )

    ###################################
    # Connect patterns and motorlayer #
    ###################################
    hind_muscle_patterns = {
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
    }

    fore_muscle_patterns = {
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

    def update_muscle_name(name: str) -> str:
        """Update muscle name format"""
        return name.replace("_", "-")

    muscles = {
        "left": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
        "right": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
    }

    for name, muscle in muscles_config["muscles"].items():
        side = muscle["side"]
        limb = muscle["limb"]
        function = muscle.get("function", "agonist")
        muscles[side][limb][function].append(
            {
                "name": join_str(name.split("_")[2:]),
                "type": muscle["type"],
                "abbrev": muscle["abbrev"],
            }
        )

    network_options = limb_circuit(
        network_options,
        join_str(("left", "fore")),
        muscles["left"]["fore"],
        fore_muscle_patterns,
        transform_mat=get_translation_matrix(off_x=-25.0, off_y=0.0)
    )

    # network_options = limb_circuit(
    #     network_options,
    #     join_str(("right", "fore")),
    #     muscles["right"]["fore"],
    #     fore_muscle_patterns,
    #     transform_mat=get_translation_matrix(off_x=25.0, off_y=0.0)
    # )

    # network_options = limb_circuit(
    #     network_options,
    #     join_str(("left", "hind")),
    #     muscles["left"]["hind"],
    #     hind_muscle_patterns,
    #     transform_mat=get_translation_matrix(
    #         off_x=-25.0, off_y=-25.0
    #     ) @ get_mirror_matrix(mirror_x=True, mirror_y=False)
    # )

    # network_options = limb_circuit(
    #     network_options,
    #     join_str(("right", "hind")),
    #     muscles["right"]["hind"],
    #     hind_muscle_patterns,
    #     transform_mat=get_translation_matrix(
    #         off_x=25.0, off_y=-25.0
    #     ) @ get_mirror_matrix(mirror_x=True, mirror_y=False)
    # )

    # rg_commissural_edges = connect_rg_commissural()
    # network_options.add_edges(rg_commissural_edges.values())

    return network_options


def generate_quadruped_circuit(
        n_iterations: int
):
    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "quadruped"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=int(n_iterations),
            timestep=1.0,
        ),
        logs=options.NetworkLogOptions(
            n_iterations=int(n_iterations),
            buffer_size=int(n_iterations),
        ),
    )
    network_options = quadruped_circuit(network_options)
    return network_options


def run_network(*args):
    network_options = args[0]

    network = PyNetwork.from_options(network_options)
    network.setup_integrator(network_options.integration)

    # data.to_file("/tmp/sim.hdf5")

    # Integrate
    N_ITERATIONS = network_options.integration.n_iterations
    states = np.ones((len(network.data.states.array),)) * 1.0

    # network_gui = NetworkGUI(data=data)
    # network_gui.run()

    inputs_view = network.data.external_inputs.array
    for iteration in tqdm(range(0, N_ITERATIONS), colour="green", ascii=" >="):
        inputs_view[:] = (iteration / N_ITERATIONS) * 1.0
        # states = rk4(iteration * 1e-3, states, network.ode, step_size=1)
        # states = network.integrator.step(network, iteration * 1e-3, states)
        network.step()
        # states = network.ode(iteration*1e-3, states)
        # print(np.array(states)[0], network.data.states.array[0], network.data.derivatives.array[0])
        network.data.times.array[iteration] = iteration*1e-3
        # network.logging(iteration)

    # network.data.to_file("/tmp/network.h5")
    network_options.save("/tmp/network_options.yaml")

    return network


def plot_network(network_options):
    """ Plot only network """

    network_options = update_edge_visuals(network_options)
    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="source",
        target="target",
    )

    # plt.figure()
    # sparse_array = nx.to_scipy_sparse_array(graph)
    # sns.heatmap(
    #     sparse_array.todense()[50:75, 50:75], cbar=False, square=True,
    #     linewidths=0.5,
    #     annot=True
    # )
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
        node_size=[300*data["visual"]["radius"] for node, data in graph.nodes.items()],
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
            node: data["visual"]["position"][:2]
            for node, data in graph.nodes.items()
        },
        edge_color=[
            [0.3, 1.0, 0.3] if data["type"] == "excitatory" else [0.7, 0.3, 0.3]
            for edge, data in graph.edges.items()
        ],
        width=1.0,
        arrowsize=10,
        style="-",
        arrows=True,
        min_source_margin=5,
        min_target_margin=5,
        connectionstyle=[
            data["visual"]["connectionstyle"]
            for edge, data in graph.edges.items()
        ],
    )
    plt.show()


def plot_data(network, network_options):
    plot_nodes = [
        index
        for index, node in enumerate(network.data.nodes)
        if ("RG_F" in node.name) and ("DR" not in node.name)
    ]

    plt.figure()
    for index, node_index in enumerate(plot_nodes):
        plt.fill_between(
            np.array(network.data.times.array),
            index + np.array(network.data.nodes[node_index].output.array),
            index,
            alpha=0.2,
            lw=1.0,
        )
        plt.plot(
            np.array(network.data.times.array),
            index + np.array(network.data.nodes[node_index].output.array),
            label=network.data.nodes[node_index].name,
        )
    plt.legend()

    plot_nodes = [
        index
        for index, node in enumerate(network.data.nodes)
        if ("Mn" in node.name)
    ]
    plt.figure()
    for index, node_index in enumerate(plot_nodes):
        plt.fill_between(
            np.array(network.data.times.array),
            index + np.array(network.data.nodes[node_index].output.array),
            index,
            alpha=0.2,
            lw=1.0,
        )
        plt.plot(
            np.array(network.data.times.array),
            index + np.array(network.data.nodes[node_index].output.array),
            label=network.data.nodes[node_index].name,
        )
    plt.legend()
    plt.show()


def main():
    """Main."""

    # Generate the network
    # network_options = generate_network(int(1e4))
    # network_options = generate_limb_circuit(int(1e4))
    network_options = generate_quadruped_circuit((5e4))
    network_options.save("/tmp/network_options.yaml")

    # Run the network
    # network = profile.profile(run_network, network_options)
    # network = run_network(network_options)

    # Results
    plot_network(network_options)

    # run_network()


if __name__ == "__main__":
    main()
