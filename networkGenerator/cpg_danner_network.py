""" Danner CPG Model. """

from network_generator import NetworkGenerator
import matplotlib.pyplot as plt
import numpy as np
import biolog
from danner_net_gen import CPG
import networkx as nx

def main():
    """Main."""

    net1 = CPG('FL', anchor_x=0., anchor_y=0.)  #: Directed graph
    net2 = CPG('FR', anchor_x=8., anchor_y=0.)  #: Directed graph
    net3 = CPG('HL', anchor_x=0., anchor_y=8.)  #: Directed graph
    net4 = CPG('HR', anchor_x=8., anchor_y=8.)  #: Directed graph    
    nx.write_graphml(nx.compose_all([net1.cpg, net2.cpg, net3.cpg, net4.cpg]),
                     './conf/auto_gen_danner_cpg.graphml')
    
    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_danner_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time = np.arange(0, 1000, dt)  #: Time
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
