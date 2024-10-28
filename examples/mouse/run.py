""" Generate and reproduce Zhang, Shevtsova, et al. eLife 2022;11:e73424. DOI:
https://doi.org/10.7554/eLife.73424 paper network """


from farms_core.io.yaml import read_yaml
from farms_network.core import options

from components import *


def generate_network(n_iterations: int):
    """Generate network"""

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "mouse"},
        integration=options.IntegrationOptions.defaults(n_iterations=n_iterations),
        logs=options.NetworkLogOptions(
            n_iterations=n_iterations,
        )
    )

    ##############
    # MotorLayer #
    ##############
    # read muscle config file
    muscles_config = read_yaml(
        "/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/muscles/quadruped_siggraph.yaml"
    )

    def update_muscle_name(name: str) -> str:
        """ Update muscle name format """
        return name.replace("_", "-")

    muscles = {
        "left": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []}
        },
        "right": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []}
        },
    }

    for name, muscle in muscles_config["muscles"].items():
        side = muscle["side"]
        limb = muscle["limb"]
        function = muscle.get("function", "agonist")
        muscles[side][limb][function].append(
            {
                "name": name,
                "type": muscle['type'],
                "abbrev": muscle['abbrev']
            }
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

            # Motor Layer
            motor_x = pf_x + 0.5*max(
                len(muscles["left"]["fore"]["agonist"]),
                len(muscles["left"]["fore"]["antagonist"])
            )
            motor_y = pf_y + 5.0
            left_fore_motor = MotorLayer(
                muscles=muscles[side][limb],
                name=f"{side}_{limb}_motor",
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((left_fore_motor.nodes()).values())

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

    return network_options


def run_network(network_options: options.NetworkOptions):


    data = NetworkData.from_options(network_options)

    network = PyNetwork.from_options(network_options)

    # data.to_file("/tmp/sim.hdf5")

    # # Integrate
    N_ITERATIONS = network_options.integration.n_iterations
    states = np.ones((len(data.states.array),)) * 1.0

    # network_gui = NetworkGUI(data=data)
    # network_gui.run()

    # for index, node in enumerate(network_options.nodes):
    #     print(index, node.name)
    inputs_view = network.data.external_inputs.array
    for iteration in tqdm(range(0, N_ITERATIONS), colour="green", ascii=" >="):
        inputs_view[:] = (iteration / N_ITERATIONS) * 1.0
        states = rk4(iteration * 1e-3, states, network.ode, step_size=1)
        # network.logging(iteration)

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
    network_options = generate_network(int(1e4))

    run_network(network_options)

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
