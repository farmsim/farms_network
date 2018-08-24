""" CPG FOUR NEURON INTEGRATE AND FIRE MODEL."""

from network_generator import NetworkGenerator
import matplotlib.pyplot as plt
import numpy as np
import biolog


def main():
    """Main."""
    #: Initialize network
    net_ = NetworkGenerator('./conf/four_neuron_cpg.graphml')

    # #: Setup the integrator
    net_.setup_integrator()

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 0.001  #: Time step
    time = np.arange(0, 2, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time), len(net_.dae.x)])

    #: Integrate the network
    biolog.info('Begin Integration!')
    for idx, _ in enumerate(time):
        res[idx] = net_.step()['xf'].full()[:, 0]

    # #: Results
    net_.visualize_network(plt)  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time, res)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
