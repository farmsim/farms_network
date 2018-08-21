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
    time = np.arange(0, 20000, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            'jit': True,
            "print_stats": False}

    # #: Setup the integrator
    net_.setup_integrator(opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')
    for idx, _ in enumerate(time):
        res[idx] = net_.step()['xf'].full()[:, 0]

    # #: Results
    net_.visualize_network(plt)  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('States Plot')
    plt.plot(time, res)
    plt.legend(tuple([leg for leg in net_.dae.x.keys()]))
    plt.grid()
    plt.show()
    # net_.generate_neurons()
    # net_.generate_network()

    # net_.show_network_sparse_matrix()  #: Print network matrix

    # #: Initialize integrator properties
    # #: pylint: disable=invalid-name
    # x0 = []
    # x0.extend([0, -60, -60, -60, -60, 0])
    # x0.extend([-60, -60, 0, -60, -60, -60])
    # x0.extend([-60, -60, -60, -60, -60, -60])
    # x0.extend([-60, 0, 0, 0, 0, 0])

    # #: Setup the integrator
    # net_.setup_integrator(x0)

    # #: Initialize network parameters
    # #: pylint: disable=invalid-name
    # dt = 1  #: Time step
    # time = np.arange(0, 500, dt)  #: Time
    # #: Vector to store results
    # res = np.empty([len(time), net_.num_states])

    # #: Integrate the network
    # biolog.info('Begin Integration!')
    # for idx, t in enumerate(time):
    #     if (t % 100) == 0:
    #         biolog.info('Time : {}'.format(t/1000.))
    #     res[idx] = net_.step([0.2])['xf'].full()[:, 0]

    # #: Results
    # net_.save_network_to_dot()  #: Save network to dot file
    # net_.visualize_network()  #: Visualize network using Matplotlib

    # plt.figure()
    # plt.title('States Plot')
    # plt.plot(time, res)
    # plt.grid()

    # plt.show()


if __name__ == '__main__':
    main()
