""" Test farms network options """


from pprint import pprint

import networkx as nx
from farms_core.io.yaml import read_yaml, write_yaml
from farms_core.options import Options
from farms_network.core import options

param_opts = options.LIDannerParameterOptions.defaults()
state_opts = options.LIDannerNaPStateOptions.from_kwargs(v0=0.0, h0=-70.0)
vis_opts = options.NodeVisualOptions()

n1_opts = options.LIDannerNodeOptions(
    name="n1",
    parameters=param_opts,
    visual=vis_opts,
    state=state_opts,
)

network = options.NetworkOptions(
    directed=True,
    multigraph=False,
    graph={"name": "network"},
    nodes=[n1_opts, n1_opts],
    edges=[],
)

print(type(network))
network.save("/tmp/opts.yaml")

pprint(options.NetworkOptions.load("/tmp/opts.yaml"))
print(type(options.NetworkOptions.load("/tmp/opts.yaml")))
