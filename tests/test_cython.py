#!/usr/bin/env python3
import pprint
from scipy.integrate import ode
from IPython import embed
from matplotlib import pyplot as plt
from farms_network_generator.network_generator import NetworkGenerator
import timeit
import numpy as np
import time
import pstats
import farms_pylog as pylog
import cProfile
setup = """
from farms_dae_generator.dae_generator import DaeGenerator
from farms_network_generator.leaky_integrator import LeakyIntegrator

d = DaeGenerator()

n1 = LeakyIntegrator('n1', d)
n2 = LeakyIntegrator('n2', d)

n1.add_ode_input(n2, **{'weight': 1.})
n2.add_ode_input(n1, **{'weight': 1.})

d.initialize_dae()
"""

# print(timeit.timeit(stmt="n1.ode_rhs();n2.ode_rhs()", setup=setup, number=1))


neurons = NetworkGenerator('./auto_gen_danner_current_openloop_opti.graphml')

neurons.initialize_dae()
pylog.warning('X0 Shape {}'.format(np.shape(neurons.dae.xdot.values)))

pylog.debug(np.array(neurons.dae.x.values))


def setup_integrator(x0, integrator='dop853', atol=1e-6,
                     rtol=1e-6, method='bdf'):
    """Setup system."""
    integrator = ode(neurons.ode).set_integrator(
        integrator,
        method=method,
        atol=atol,
        rtol=rtol)
    integrator.set_initial_value(x0, 0.0)
    return integrator


integrator = setup_integrator(neurons.dae.x.values,
                              integrator='lsoda', atol=1e-3,
                              rtol=1e-3)

pylog.debug("Number of states {}".format(len(neurons.dae.x.values)))
pylog.debug("Number of state derivatives {}".format(
    len(neurons.dae.xdot.values)))
pylog.debug("Number of parameters {}".format(len(neurons.dae.p.values)))
pylog.debug("Number of inputs {}".format(len(neurons.dae.u.values)))
pylog.debug("Number of outputs {}".format(len(neurons.dae.y.values)))

N = 6000
# neurons.dae.u.values = np.array([1.0, 0.5], dtype=np.float)
# print("Time {} : state {}".format(j, neurons.dae.y.values))

u = np.ones(np.shape(neurons.dae.u.values))


def main():
    start = time.time()
    for j in range(0, N):
        neurons.dae.u.values = u*j/N
        integrator.set_initial_value(integrator.y,
                                     integrator.t)
        neurons.dae.x.values = integrator.integrate(integrator.t+1)
        neurons.dae.update_log()
    end = time.time()
    print('TIME {}'.format(end-start))


# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("cumtime").print_stats()
# cProfile.run("main()", "simulation.profile")
# s.strip_dirs().print_stats()
cProfile.runctx("main()",
                globals(), locals(), "Profile.prof")
pstat = pstats.Stats("Profile.prof")
pstat.sort_stats('time').print_stats()
pstat.sort_stats('cumtime').print_stats()

# data_x = neurons.dae.x.log
# plt.title('X')
# plt.plot(np.linspace(0, N*0.001, N), data_x[:N, :])
# plt.legend(tuple([str(key) for key in range(len(neurons.dae.x.values))]))
# plt.grid(True)

# plt.figure(3)
# plt.title('Y')
# data_y = neurons.dae.y.log
# plt.plot(np.linspace(0, N*0.001, N), data_y[:N, :])
# plt.grid(True)
# plt.show()
