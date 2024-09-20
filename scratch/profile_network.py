""" Profile network implementation """


import numpy as np
from farms_core.utils.profile import profile
from farms_network.core import network
from farms_network.core.data import NetworkData, StatesArray
from farms_network.models import li_danner

nstates = 100
niterations = int(100e3)
states = StatesArray(np.empty((niterations, nstates)))

data = NetworkData(nstates=100, states=states)

net = network.PyNetwork(nneurons=100)
profile(net.test, data)
