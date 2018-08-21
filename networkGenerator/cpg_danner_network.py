""" Danner CPG Model. """

from network_generator import NetworkGenerator
import matplotlib.pyplot as plt
import numpy as np
import biolog


def main():
    """Main."""
    #: Initialize network
    net_ = NetworkGenerator('./conf/simple_danner_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time = np.arange(0, 5000, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            'jit': False,
            "print_stats": False}

    # #: Setup the integrator
    net_.setup_integrator(opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')
    for idx, _ in enumerate(time):
        net_.dae.u.set_all(0.2)
        res[idx] = net_.step()['xf'].full()[:, 0]

    # #: Results
    net_.visualize_network(plt)  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time, res)
    plt.legend(tuple([leg for leg in net_.dae.x.keys()]))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
