""" Test network """

import numpy as np
from farms_network.core.network import PyNetwork
from farms_network.core.options import NetworkOptions
from scipy.integrate import ode

nstates = 100
network = PyNetwork.from_options(nnodes=nstates)

integrator = ode(network.ode).set_integrator(
    u'dopri5',
    method=u'adams',
    max_step=0.0,
)
integrator.set_initial_value(np.zeros((nstates,)), 0.0)


# # Integrate
# for iteration in range(0, 100):
#     integrator.set_initial_value(integrator.y, integrator.t)
#     integrator.integrate(integrator.t+(iteration*1e-3))
