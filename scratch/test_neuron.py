import numpy as np
from farms_network.core import li_danner, network, neuron, options
from farms_network.data.data import NetworkData, StatesArray


nstates = 100
niterations = 1000
states = StatesArray(np.empty((niterations, nstates)))

data = NetworkData(nstates=100, states=states)


net = network.PyNetwork(nneurons=10)
net.test(data)

n1_opts = options.NeuronOptions(
    name="n1",
    parameters=options.NeuronParameterOptions(),
    visual=options.NeuronVisualOptions(),
    state=options.NeuronStateOptions(initial=[0, 0]),
)
n1 = neuron.PyNeuron.from_options(n1_opts)
n1_opts.save("/tmp/opts.yaml")


print(n1.name)
n1.name = "n2"
print(n1.model_type)
print(n1.name)

states = np.empty((1,))
dstates = np.empty((1,))
inputs = np.empty((10,))
weights = np.empty((10,))
noise = np.empty((10,))
drive = 0.0

print(
    n1.ode_rhs(0.0, states, dstates, inputs, weights, noise, drive)
)

print(
    n1.output(0.0, states)
)

n2 = li_danner.PyLIDannerNeuron("n2", ninputs=50)

print(n2.name)
print(n2.model_type)
n2.name = "n2"
print(n2.name)

states = np.empty((1,))
dstates = np.empty((1,))
inputs = np.empty((10,))
weights = np.empty((10,))
noise = np.empty((10,))
drive = 0.0

print(
    n2.output(0.0, states)
)
