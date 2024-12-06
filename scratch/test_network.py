""" Test network """

from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from farms_core.io.yaml import read_yaml, write_yaml
from farms_core.options import Options
from farms_network.core import options
from farms_network.core.data import (NetworkConnectivity, NetworkData,
                                     NetworkStates)
from farms_network.core.network import PyNetwork, rk4
from farms_network.core.options import NetworkOptions
from scipy.integrate import ode
from tqdm import tqdm


def linear_network():
    """ Linear stateless network """
    param_opts = options.LinearParameterOptions.defaults()
    vis_opts = options.NodeVisualOptions()

    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "network"},
    )

    network_options.add_node(
        options.LinearNodeOptions(
            name="node1",
            parameters=param_opts,
            visual=vis_opts,
        )
    )
    return network_options


def quadruped_network():
    """ Quadruped network """
    param_opts = options.LIDannerParameterOptions.defaults()
    state_opts = options.LINaPDannerStateOptions.from_kwargs(v0=0.0, h0=-70.0)
    vis_opts = options.NodeVisualOptions()

    danner_network = nx.read_graphml(
        "/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/network/siggraph_network.graphml"
    )

    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "network"},
    )

    for node, data in danner_network.nodes.items():
        if data["model"] == "li_nap_danner":
            network_options.add_node(
                options.LIDannerNodeOptions(
                    name=node,
                    parameters=param_opts,
                    visual=vis_opts,
                    state=state_opts,
                )
            )
        else:
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
    return network_options


def oscillator_network():
    """ Oscillator network """

    param_opts = options.OscillatorNodeParameterOptions.defaults()
    state_opts = options.OscillatorStateOptions.from_kwargs(phase=0.0, amplitude=0.0)
    vis_opts = options.NodeVisualOptions()

    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "network"},
    )

    network_options.add_node(
        options.OscillatorNodeOptions(
            name="O1",
            parameters=param_opts,
            visual=vis_opts,
            state=state_opts,
        )
    )

    network_options.add_node(
        options.OscillatorNodeOptions(
            name="O2",
            parameters=param_opts,
            visual=vis_opts,
            state=state_opts,
        )
    )

    network_options.add_edge(
        options.EdgeOptions(
            source="O1",
            target="O2",
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
            parameters=options.OscillatorEdgeParameterOptions(
                phase_difference=np.pi/2
            )
        )
    )

    network_options.add_edge(
        options.EdgeOptions(
            source="O2",
            target="O1",
            weight=1.0,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
            parameters=options.OscillatorEdgeParameterOptions(
                phase_difference=-np.pi/4
            )
        )
    )

    return network_options

network_options = linear_network()

network_options = oscillator_network()
# network_options = quadruped_network()

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
iterations = 10000
states = np.ones((iterations+1, len(data.states.array)))*1.0
outputs = np.ones((iterations, len(data.outputs.array)))*1.0
# states[0, 2] = -1.0

for iteration in tqdm(range(0, iterations), colour='green', ascii=' >='):
    # integrator.set_initial_value(integrator.y, integrator.t)
    # states[iteration+1, :] = states[iteration, :] + np.array(network.ode(iteration*1e-3, states[iteration, :]))*1e-3
    # network.ode(0.0, states)
    # network.ode(0.0, states)
    # network.ode(0.0, states)
    network.data.external_inputs.array[:] = np.ones((1,))*np.sin(iteration*1e-3)
    states[iteration+1, :] = rk4(iteration*1e-3, states[iteration, :], network.ode, step_size=1)
    outputs[iteration, :] = network.data.outputs.array

    # integrator.integrate(integrator.t+(iteration*1e-3))

plt.plot(np.linspace(0.0, iterations*1e-3, iterations), outputs[:, :])
plt.show()
