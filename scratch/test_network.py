""" Test network """

import numpy as np
from farms_network.core.network import PyNetwork
from farms_network.core.options import NetworkOptions
from scipy.integrate import ode

nnodes = 100

states = np.zeros((nnodes,))
dstates = np.zeros((nnodes,))
outputs = np.zeros((nnodes,))
weights = np.zeros((nnodes,))

network = PyNetwork(nnodes=nnodes)

integrator = ode(network.ode).set_integrator(
    u'dopri5',
    method=u'adams',
    max_step=0.0,
    nsteps=4
)
integrator.set_initial_value(np.zeros((nnodes,)), 0.0)

integrator.integrate(integrator.t+1.0)

# # Integrate
# for iteration in range(0, 100):
#     integrator.set_initial_value(integrator.y, integrator.t)
#     integrator.integrate(integrator.t+(iteration*1e-3))
