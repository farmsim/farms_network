#!/usr/bin/env python3
import pprint
from scipy.integrate import ode
from IPython import embed
from matplotlib import pyplot as plt
from farms_network_generator.neural_system import NeuralSystem
from farms_dae_generator.dae_generator import DaeGenerator
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

neural = NeuralSystem('./auto_gen_danner_current_openloop_opti.graphml')

neural.setup_integrator()

pylog.warning('X0 Shape {}'.format(np.shape(neural.dae.xdot.values)))

pylog.debug(np.array(neural.dae.x.values))

pylog.debug("Number of states {}".format(len(neural.dae.x.values)))
pylog.debug("Number of state derivatives {}".format(
    len(neural.dae.xdot.values)))
pylog.debug("Number of parameters {}".format(len(neural.dae.p.values)))
pylog.debug("Number of inputs {}".format(len(neural.dae.u.values)))
pylog.debug("Number of outputs {}".format(len(neural.dae.y.values)))

N = 5000
# dae.u.values = np.array([1.0, 0.5], dtype=np.float)
# print("Time {} : state {}".format(j, dae.y.values))


u = np.ones(np.shape(neural.dae.u.values))


def main():
    start = time.time()
    for j in range(0, N):
        # dae.u.values = u*j/N
        neural.step(dt=1)
    end = time.time()
    print('TIME {}'.format(end-start))


# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("cumtime").print_stats()
# cProfile.run("main()", "simulation.profile")
# s.strip_dirs().print_stats()
cProfile.runctx("main()",
                globals(), locals(), "Profile.prof")
pstat = pstats.Stats("Profile.prof")
# pstat.sort_stats('time').print_stats()
# pstat.sort_stats('cumtime').print_stats()

data_x = neural.dae.x.log
plt.title('X')
plt.plot(np.linspace(0, N*0.001, N), data_x[:N, -1])
plt.legend(tuple([str(key) for key in range(len(neural.dae.x.values))]))
plt.grid(True)

# plt.figure(3)
# plt.title('Y')
neural_y = neural.dae.y.log
neural_names = neural.dae.y.names
# plt.plot(np.linspace(0, N*0.001, N), np.sin(data_y[:N, :]))
# plt.grid(True)
# plt.show()

plt.figure()
plt.title('neural_y')
_names = []
for j, name in enumerate(neural_names):
    if 'Mn' in name:
        if ('HL' in name) or ('HR' in name):
            print(j, name)
            _names.append(j)
plt.plot(neural_y[_names], '*')
plt.legend(tuple(_names))
plt.grid(True)
plt.show()
