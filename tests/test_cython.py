#!/usr/bin/env python3
from matplotlib import pyplot as plt
from farms_network_generator.network_generator import NetworkGenerator
import timeit
import numpy as np
import time
import pstats
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


neurons = NetworkGenerator('./four_neuron_cpg.graphml')
# x0 = np.array([1, 0.5, 1, 0.5], dtype=np.float64)
# neurons.setup_integrator(x0, integrator='dopri5',
#                          atol=1e-3,
#                          rtol=1e-3)
# print(neurons.dae.x.log)
# N = 100000
# # neurons.dae.u.values = np.array([1.0, 0.5], dtype=np.float)
# # print("Time {} : state {}".format(j, neurons.dae.y.values))

# # np.ones((4,))*np.sin(2*3.14*1.*time)


# def main():
#     start = time.time()
#     for j in range(0, N):
#         u = np.ones((4,))*np.sin(2*3.14*1.*j*0.001)
#         neurons.step(u)
#     end = time.time()
#     print('TIME {}'.format(end-start))


# # s = pstats.Stats("Profile.prof")
# # s.strip_dirs().sort_stats("cumtime").print_stats()
# # cProfile.run("main()", "simulation.profile")
# cProfile.runctx("main()",
#                 globals(), locals(), "Profile.prof")
# pstat = pstats.Stats("Profile.prof")
# pstat.sort_stats('time').print_stats(30)
# pstat.sort_stats('cumtime').print_stats(30)
# # s.strip_dirs().print_stats()


# # for j in range(0, N):
# #     neurons.step()
# print('Parameters {}'.format(neurons.dae.p.log))
# print('Constants {}'.format(neurons.dae.c.log))

# data_x = neurons.dae.x.log
# plt.plot(np.linspace(0, N*0.001, N), data_x[:N, :])
# plt.plot(np.linspace(0, N*0.001, N), data_x[:N, :], 'o', markersize=0.5)
# plt.grid(True)
# plt.show()
