""" Test network """

from copy import deepcopy
from pprint import pprint

import networkx as nx
import numpy as np
from farms_core.io.yaml import read_yaml, write_yaml
from farms_core.options import Options
from farms_network.core import options
from farms_network.core.data import (NetworkConnectivity, NetworkData,
                                     NetworkStates)
from farms_network.core.network import PyNetwork
from farms_network.core.options import NetworkOptions
from scipy.integrate import ode
from tqdm import tqdm

param_opts = options.LIDannerParameterOptions.defaults()
state_opts = options.LIDannerNaPStateOptions.from_kwargs(v0=0.0, h0=-70.0)
vis_opts = options.NodeVisualOptions()

danner_network = nx.read_graphml("/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/network/siggraph_network.graphml")

network_options = options.NetworkOptions(
    directed=True,
    multigraph=False,
    graph={"name": "network"},
)

for node in danner_network.nodes:
    network_options.add_node(
        options.LIDannerNodeOptions(
            name=node,
            parameters=param_opts,
            visual=vis_opts,
            state=state_opts,
        )
    )

for edge, data in danner_network.edges.items():
    network_options.add_edge(
        options.EdgeOptions(
            source=edge[0],
            target=edge[1],
            weight=data["weight"],
            type=data.get("type", "excitatory"),
            visual=options.EdgeVisualOptions(),
        )
    )

data = NetworkData.from_options(network_options)

network = PyNetwork.from_options(network_options)

integrator = ode(network.ode).set_integrator(
    u'dopri5',
    method=u'adams',
    max_step=0.0,
    nsteps=0
)
# nnodes = len(network_options.nodes)
# integrator.set_initial_value(np.zeros(len(data.states.array),), 0.0)

# print("Data ------------", np.array(network.data.states.array))

# data.to_file("/tmp/sim.hdf5")

# integrator.integrate(integrator.t + 1e-3)

# # Integrate
states = np.array(np.arange(0, len(data.states.array)), dtype=np.double)
network.ode(0.0, states)
for iteration in tqdm(range(0, 100000), colour='green'):
    # integrator.set_initial_value(integrator.y, integrator.t)
    network.ode(0.0, states)
    # integrator.integrate(integrator.t+(iteration*1e-3))
