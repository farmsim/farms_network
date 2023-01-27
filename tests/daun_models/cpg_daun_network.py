#!/usr/bin/env python

""" Dauns CPG Model. """

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from farms_network.neural_system import NeuralSystem
from farms_container import Container

import farms_pylog as pylog
from daun_net_gen import (CPG, ConnectCPG2Interneurons,
                          Interneurons, SideNetwork)

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

TIME_SCALE = 1
TIME_UNITS = 0.001


def main():
    """Main."""

    #: Left side network
    net_right = SideNetwork('R', 30.0, 0.0)
    net_left = SideNetwork('L', 0.0, 0.0)

    # net1 = CPG('PR_L1', anchor_x=0, anchor_y=0)
    # net2 = Interneurons('PR_L1', anchor_x=0., anchor_y=0.)

    # net_C_IN_1 = ConnectCPG2Interneurons(net1.cpg, net2.interneurons)

    # net = nx.compose_all([net_C_IN_1.net])

    net = nx.compose_all([net_right.net, net_left.net])

    #: Location to save the network
    net_dir = os.path.join(
        os.path.dirname(__file__),
        './conf/auto_gen_daun_cpg.graphml')
    try:
        nx.write_graphml(net, net_dir)
    except IOError:
        if not os.path.isdir(os.path.split(net_dir)[0]):
            pylog.info('Creating directory : {}'.format(net_dir))
            os.mkdir(os.path.split(net_dir)[0])
            nx.write_graphml(net, net_dir)
        else:
            pylog.error('Error in creating directory!')
            raise IOError()

    #: Initialize network
    dt = 10  #: Time step
    dur = 5000
    time_vec = np.arange(0, dur, dt)  #: Time

    container = Container(dur/dt)
    net_ = NeuralSystem('./conf/auto_gen_daun_cpg.graphml', container)

    #: initialize network parameters
    #: pylint: disable=invalid-name

    #: initialize network parameters
    container.initialize()
    x0 = [-70 if 'V_' in name else 0.0 for name in container.neural.states.names]
    net_.setup_integrator()

    #: Integrate the network
    pylog.info('Begin Integration!')
    start_time = time.time()
    for idx, _ in enumerate(time_vec):
        net_.step(dt=dt)
        container.update_log()
    end_time = time.time()

    pylog.info('Execution Time : {}'.format(
        end_time - start_time))

    # #: Results
    net_.save_network_to_dot()
    #: Visualize network using Matplotlib
    net_.visualize_network(plt_out=plt)
    plt.figure()
    plt.title('DAUNS NETWORK')
    outputs = container.neural.outputs
    plt.plot(time_vec*TIME_UNITS, outputs.log)
    # plt.plot(time_vec*TIME_UNITS,
    #          outputs[:, container.neural.outputs.get_parameter_index('nout_'+'PR_L1_C2')])
    plt.xlabel('Time [s]')
    # plt.legend(('V_PR_L1_C1', 'V_PR_L1_C2'))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
