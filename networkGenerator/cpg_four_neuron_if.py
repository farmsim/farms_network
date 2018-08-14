""" CPG FOUR NEURON INTEGRATE AND FIRE MODEL."""

from neuralnet import NeuralNetGen
import matplotlib.pyplot as plt
import numpy as np
import biolog


def main():
    """Main."""
    #: Initialize network
    net_ = NeuralNetGen('./conf/four_neuron_cpg.graphml')
    net_.generate_neurons()
    net_.generate_network()

    #: Set neural properties
    net_.neurons['N1'].bias = 3.0
    net_.neurons['N2'].bias = 3.0
    net_.neurons['N3'].bias = -3.0
    net_.neurons['N4'].bias = -3.0

    net_.neurons['N1'].tau = 0.02
    net_.neurons['N2'].tau = 0.02
    net_.neurons['N3'].tau = 0.1
    net_.neurons['N4'].tau = 0.1
    net_.show_network_sparse_matrix()  #: Print network matrix

    #: Initialize integrator properties
    #: pylint: disable=invalid-name
    x0 = [5, 2, 5, 2]  #: Neuron 1 and 2 membrane potentials

    #: Setup the integrator
    net_.setup_integrator(x0)

    #: Initialize network parameters
    #: pylint: disable=invalid-name
    dt = 0.01  #: Time step
    time = np.arange(0, 2, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time), net_.num_states])

    #: Integrate the network
    biolog.info('Begin Integration!')
    for idx, _ in enumerate(time):
        res[idx] = net_.step()['xf'].full()[:, 0]

    #: Results
    net_.save_network_to_dot()  #: Save network to dot file
    net_.visualize_network()  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time, res)
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
