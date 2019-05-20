#!/usr/bin/env python3
from matplotlib import pyplot as plt
from farms_network_generator.network_generator import NetworkGenerator
import timeit
import numpy as np
import time
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


neurons = NetworkGenerator(
    '/home/tatarama/Stata-PhD/Projects/BioRobAnimals/network_generator/tests/integrate_fire/conf/four_neuron_cpg.graphml')
x0 = np.random.random((4, 1))
neurons.setup_integrator(x0)
print(neurons.dae.x.log)
start = time.time()
N = 100000
for j in range(0, N):
    # neurons.dae.u.values = np.array([1.0, 0.5], dtype=np.float)
    neurons.step()
    # print("Time {} : state {}".format(j, neurons.dae.y.values))
end = time.time()
print('TIME {}'.format(end-start))
print(neurons.dae.p.values)
data_x = neurons.dae.x.log
plt.plot(np.linspace(0, N*0.001, N), data_x[:N, :])
plt.show()
