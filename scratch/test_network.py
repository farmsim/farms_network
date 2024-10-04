""" Test network """

from pprint import pprint

from copy import deepcopy
import networkx as nx
import numpy as np
from farms_core.io.yaml import read_yaml, write_yaml
from farms_core.options import Options
from farms_network.core import options
from farms_network.core.network import PyNetwork
from farms_network.core.options import NetworkOptions
from scipy.integrate import ode
from farms_network.core.data import NetworkData, NetworkConnectivity, NetworkStates


param_opts = options.LIDannerParameterOptions.defaults()
state_opts = options.LIDannerNaPStateOptions.from_kwargs(v0=0.0, h0=-70.0)
vis_opts = options.NodeVisualOptions()

n1_opts = options.LIDannerNodeOptions(
    name="n1",
    parameters=param_opts,
    visual=vis_opts,
    state=state_opts,
)

n2_opts = deepcopy(n1_opts)
n2_opts.name = "n2"

edge_n1_n2 = options.EdgeOptions(
    source="n1",
    target="n2",
    weight=1.0,
    type="excitatory",
    visual=options.EdgeVisualOptions(),
)

edge_n2_n1 = options.EdgeOptions(
    source="n2",
    target="n1",
    weight=1.0,
    type="excitatory",
    visual=options.EdgeVisualOptions(),
)

network = options.NetworkOptions(
    directed=True,
    multigraph=False,
    graph={"name": "network"},
)

network.add_node(n1_opts)
network.add_node(n2_opts)
network.add_edge(edge_n1_n2)
network.add_edge(edge_n2_n1)

nnodes = 10

danner_network = nx.read_graphml("/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/network/siggraph_network.graphml")

network_options = options.NetworkOptions(
    directed=True,
    multigraph=False,
    graph={"name": "network"},
)

for node in danner_network.nodes:
    network.add_node(
        options.LIDannerNodeOptions(
            name=node,
            parameters=param_opts,
            visual=vis_opts,
            state=state_opts,
        )
    )

for edge in danner_network.edges:
    network.add_edge(
        options.EdgeOptions(
            source=edge[0],
            target=edge[1],
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
    )

data = NetworkData.from_options(network_options)

network = PyNetwork.from_options(network)

integrator = ode(network.ode).set_integrator(
    u'dopri5',
    method=u'adams',
    max_step=0.0,
    nsteps=0
)
nnodes = len(network_options.nodes)
integrator.set_initial_value(np.zeros((nnodes,)), 0.0)

print(network.nnodes)

data.to_file("/tmp/sim.hdf5")

# integrator.integrate(integrator.t)

# # Integrate
# for iteration in range(0, 100):
#     integrator.set_initial_value(integrator.y, integrator.t)
#     integrator.integrate(integrator.t+(iteration*1e-3))
