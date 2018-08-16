""" Danner CPG Model. """

from neuralnet import NeuralNetGen
import matplotlib.pyplot as plt
import numpy as np
import biolog


def main():
    """Main."""
    #: Initialize network
    net_ = NeuralNetGen('./conf/simple_daun_cpg.graphml')
    net_.generate_neurons()
    net_.generate_network()

    #: Initialize integrator properties
    #: pylint: disable=invalid-name
    x0 = []
    x0.extend([-80.0, 0.0, -60.0, 0.0])

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
    for idx, t in enumerate(time):
        _out = net_.step(params=[t*0.05*1e-3])
        res[idx] = _out['xf'].full()[:, 0]
        res_deep[idx] = _out['zf'].full()[:, 0]

    #: Results
    net_.save_network_to_dot()  #: Save network to dot file
    net_.visualize_network()  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time, res[:, [0, 2]])
    plt.legend(('C2', 'C1'))
    plt.grid()

    plt.figure()
    plt.title('Synapse')
    plt.plot(time, res_deep)
    plt.legend(('C2', 'C1'))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
