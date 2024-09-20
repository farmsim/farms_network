import numpy as np
from farms_network.core.data import NetworkData, StatesArray

nstates = 100
niterations = 1000
states = StatesArray(
    np.empty((niterations, nstates))
)

data = NetworkData(nstates=100, states=states)

print(data.states.array[0, 0])
