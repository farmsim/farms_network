""" Danner CPG Model. """

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import biolog
from network_models.daun_cpg.daun_net_gen import SideNetwork
from network_generator.network_generator import NetworkGenerator

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def main():
    """Main."""

    #: Left side network
    net_right = SideNetwork('R', 30.0, 0.0)
    net_left = SideNetwork('L', 0.0, 0.0)

    net = nx.compose_all([net_left.net,
                          net_right.net])

    #: Location to save the network
    net_dir = os.path.join(
        os.path.dirname(__file__),
        './conf/auto_gen_daun_cpg.graphml')
    try:
        nx.write_graphml(net, net_dir)
    except IOError:
        if not os.path.isdir(os.path.split(net_dir)[0]):
            biolog.info('Creating directory : {}'.format(net_dir))
            os.mkdir(os.path.split(net_dir)[0])
            nx.write_graphml(net, net_dir)
        else:
            biolog.error('Error in creating directory!')
            raise IOError()

    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_daun_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time_vec = np.arange(0, 10000, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time_vec), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            'jit': False,
            "enable_jacobian": True,
            "print_time": False,
            "print_stats": False,
            "reltol": 1e-6,
            "abstol": 1e-6}

    # #: Setup the integrator
    net_.setup_integrator(integration_method='cvodes', opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')

    biolog.info('PARAMETERS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.p.param_list]))

    biolog.info('INPUTS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.u.param_list]))

    biolog.info('INITIAL CONDITIONS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.x.param_list]))

    biolog.info('CONSTANTS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.c.param_list]))

    start_time = time.time()
    for idx, _ in enumerate(time_vec):
        res[idx] = net_.step()['xf'].full()[:, 0]
    end_time = time.time()

    biolog.info('Execution Time : {}'.format(
        end_time - start_time))

    # #: Results
    net_.save_network_to_dot()
    #: Visualize network using Matplotlib
    net_.visualize_network(plt_out=plt)
    plt.figure()
    plt.title('DAUNS NETWORK')
    plt.plot(time_vec*0.001,
             res[:, [net_.dae.x.get_idx('V_PR_L1_C1')]])
    plt.plot(time_vec*0.001,
             res[:, [net_.dae.x.get_idx('V_PR_L1_C2')]],
             ':', markersize=5.)
    plt.xlabel('Time [s]')
    plt.legend(('V_PR_L1_C1', 'V_PR_L1_C2'))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
