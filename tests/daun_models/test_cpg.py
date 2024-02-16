#!/usr/bin/env python

"""Test CPG network"""

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

    cpg = CPG('cpg', anchor_x=0, anchor_y=0)

    net = nx.compose_all([cpg.cpg])

    # Location to save the network
    net_dir = os.path.join(
        os.path.dirname(__file__),
        './conf/auto_gen_daun_cpg_test.graphml')
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

    # Initialize network
    dt = 1  # Time step
    dur = 5000
    time_vec = np.arange(0, dur, dt)  # Time

    container = Container(dur/dt)
    net_ = NeuralSystem('./conf/auto_gen_daun_cpg_test.graphml', container)

    # initialize network parameters
    # pylint: disable=invalid-name

    # initialize network parameters
    container.initialize()
    x0 = [-10.0, 0.1, -70.0, 0.8]
    net_.setup_integrator()

    # Integrate the network
    pylog.info('Begin Integration!')
    start_time = time.time()
    for idx, _ in enumerate(time_vec):
        net_.step(dt=dt)
        container.update_log()
    end_time = time.time()

    pylog.info('Execution Time : {}'.format(
        end_time - start_time))

    # # Results
    net_.save_network_to_dot()
    # Visualize network using Matplotlib
    net_.visualize_network(plt_out=plt)
    plt.figure()
    plt.title('DAUNS NETWORK')
    states = container.neural.states
    plt.plot(time_vec*TIME_UNITS, states.log)
    plt.xlabel('Time [s]')
    plt.legend(states.names)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
