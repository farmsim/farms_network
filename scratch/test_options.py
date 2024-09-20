""" Test farms network options """


from pprint import pprint

import networkx as nx
from farms_core.io.yaml import read_yaml, write_yaml
from farms_core.options import Options
from farms_network.core import options

param_opts = options.LIDannerParameterOptions.defaults()
state_opts = options.LIDannerStateOptions.from_kwargs(v0=0.0, h0=1.0)
vis_opts = options.NeuronVisualOptions()

n1_opts = options.LIDannerNeuronOptions(
    name="n1",
    parameters=param_opts,
    visual=vis_opts,
    state=state_opts,
    ninputs=10,
)

network = options.NetworkOptions(
    directed=True,
    multigraph=False,
    name="network",
    neurons=[n1_opts, n1_opts],
    connections=[],
)

print(type(network))
network.save("/tmp/opts.yaml")

pprint(options.NetworkOptions.load("/tmp/opts.yaml"))
print(type(options.NetworkOptions.load("/tmp/opts.yaml")))
