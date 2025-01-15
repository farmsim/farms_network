import networkx as nx
from farms_core.analysis import plot


def visualize(network_options):
    """ Visualize network """

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
            node: data["visual"]["position"][:2]
            for node, data in graph.nodes.items()
        },
        node_color=[
            data["visual"]["color"]
            for node, data in graph.nodes.items()
        ],
        alpha=0.25,
        edgecolors="k",
        linewidths=2.0,
        node_size=[
            300*data["visual"]["radius"]
            for node, data in graph.nodes.items()
        ],
    )
    nx.draw_networkx_labels(
        graph,
        pos={
            node: data["visual"]["position"][:2]
            for node, data in graph.nodes.items()
        },
        labels={
            node: data["visual"]["label"]
            for node, data in graph.nodes.items()
        },
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
