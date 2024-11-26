import matplotlib.pyplot as plt
import numpy as np
from farms_core.io.yaml import read_yaml
from farms_network.core import options
from farms_network.noise.ornstein_uhlenbeck import OrnsteinUhlenbeck

network_options = options.NetworkOptions.from_options(
    read_yaml("/tmp/network_options.yaml")
)

n_dim = 0
for node in network_options.nodes:
    if node.noise is not None:
        if (node.noise.model == "ornstein_uhlenbeck") and node.noise.is_stochastic:
            n_dim += 1



timestep = 1e-3
tau = 1.0
sigma = 1# np.sqrt(2.0)

noise_options = [network_options.nodes[1].noise,]

oo = OrnsteinUhlenbeck(noise_options)

times = np.linspace(0, 100000*timestep, int(10000))
print(np.sqrt(2.0*timestep))

for initial, mean in zip((10.0, 0.0, -10.0, 0.0), (0.0, 0.0, 0.0, -10.0)):
    states = np.zeros((len(times), 1))
    states[0, 0] = initial
    noise_options[0].seed = np.random.randint(low=0, high=10000)
    noise_options[0].mu = mean
    noise_options[0].tau = tau
    noise_options[0].sigma = sigma
    print(noise_options)
    oo = OrnsteinUhlenbeck(noise_options)
    drift = np.zeros((len(times), 1))
    diffusion = np.zeros((len(times), 1))
    for index, time in enumerate(times[:-1]):
        drift[index, :] = oo.py_evaluate_a(time, states[index, :], drift[index, :])
        diffusion[index, :] = oo.py_evaluate_b(time, states[index, :], diffusion[index, :])
        states[index+1, :] = states[index, :] + drift[index, :]*timestep + np.sqrt(timestep)*diffusion[index, :]
    print(np.std(states[500:, 0]), np.mean(states[500:, 0]))
    plt.plot(times, states[:, 0])
plt.xlim([0, times[-1]])
plt.ylim([-15.0, 15.0])
plt.show()
