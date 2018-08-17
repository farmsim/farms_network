""" Danner CPG Model. """

from neuralnet import NeuralNetGen
import matplotlib.pyplot as plt
import numpy as np
import biolog


def main():
    """Main."""
    #: Initialize network
    net_ = NeuralNetGen('./conf/motorneuron_daun_test.graphml')
    net_.generate_neurons()
    net_.generate_network()

    #: Initialize integrator properties
    #: pylint: disable=invalid-name
    x0 = []
    x0.extend([-63.46, 0.7910,
               -65.0, 0.9, 0.0, 0.0, 0.0,
               -65.0, 0.9, 0.0, 0.0, 0.0,
               -63.46, 0.7910,
               -63.46, 0.7910,
               -63.46, 0.7910,
               -63.46487, 0.8,
               -10.0, 0.2592])

    #: Setup the integrator
    net_.setup_integrator(x0)

    #: Initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time = np.arange(0, 10000, dt)  #: Time

    #: Integrate the network
    biolog.info('Begin Integration!')
    #: Vector to store results
    res = np.empty([len(time), net_.num_states])
    res_deep = np.empty([len(time), net_.num_alg_var])
    #: Integrate the network
    for idx, _ in enumerate(time):
        _out = net_.step(params=[0.2, 0.0,
                                 0.2, 0.0,
                                 0.2, 0.0,
                                 0.2, 0.0,
                                 0.2, 0.0,
                                 0.2, 0.0,
                                 0.2, 0.0,
                                 0.2, 0.0])
        res[idx] = _out['xf'].full()[:, 0]
        res_deep[idx] = _out['zf'].full()[:, 0]

    #: Results
    net_.save_network_to_dot()  #: Save network to dot file
    net_.visualize_network()  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time/1000., res[:, [0, 2, 7, 12, 14, 16, 18, 20]])
    plt.xlabel('Time[s]')
    plt.ylabel('Membrane potential [mV]')
    plt.legend(('C2', 'C1', 'In1', 'Mn1'))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
