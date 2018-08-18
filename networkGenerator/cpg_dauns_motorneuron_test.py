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
    x0 = {}
    x0['C1'] = [-63.46487, 0.8]
    x0['C2'] = [-10.0, 0.2592]
    x0['In1'] = [-63.46, 0.7910]
    x0['In2'] = [-63.46, 0.7910]
    x0['In3'] = [-63.46, 0.7910]
    x0['In4'] = [-63.46, 0.7910]
    x0['In5'] = [-63.46, 0.7910]
    x0['In6'] = [-63.46, 0.7910]
    x0['Mn1'] = [-65.0, 0.9, 0.0, 0.0, 0.0]
    x0['Mn2'] = [-65.0, 0.9, 0.0, 0.0, 0.0]
    x0['Mn3'] = [-65.0, 0.9, 0.0, 0.0, 0.0]
    x0['Mn4'] = [-65.0, 0.9, 0.0, 0.0, 0.0]

    #: Setup the integrator
    net_.set_init_states(x0)
    net_.setup_integrator()

    #: Initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time = np.arange(0, 1000, dt)  #: Time

    #: Integrate the network
    biolog.info('Begin Integration!')
    #: Vector to store results
    res = np.empty([len(time), net_.num_states])
    res_deep = np.empty([len(time), net_.num_alg_var])

    #: set parameters
    params = {}
    params['C1'] = [0.2, 0.0]
    params['C2'] = [0.2, 0.0]
    params['In1'] = [1.6, -80.0]
    params['In2'] = [1.6, -80.0]
    params['In3'] = [1.6, -80.0]
    params['In4'] = [1.6, -80.0]
    params['In5'] = [0.0, 0.0]
    params['In6'] = [0.0, 0.0]
    params['Mn1'] = [0.19, 0.0]
    params['Mn2'] = [1.6, -80]
    params['Mn3'] = [0.19, 0.0]
    params['Mn4'] = [0.19, 0.0]
    
    #: Integrate the network
    _params = net_.set_params(params)
    for idx, _ in enumerate(time):
        _out = net_.step(params=_params)
        res[idx] = _out['xf'].full()[:, 0]
        res_deep[idx] = _out['zf'].full()[:, 0]

    #: Results
    net_.save_network_to_dot()  #: Save network to dot file
    net_.visualize_network()  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time/1000., res[:, [0, 2, 4, 6, 11, 16, 18, 20, 22,
                                 27, 29, 31]])
    plt.xlabel('Time[s]')
    plt.ylabel('Membrane potential [mV]')
    plt.legend(tuple([neuron for neuron in net_.neurons.keys()]))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
