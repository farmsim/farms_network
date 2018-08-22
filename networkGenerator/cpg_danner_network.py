""" Danner CPG Model. """

from network_generator import NetworkGenerator
import matplotlib.pyplot as plt
import numpy as np
import biolog
from danner_net_gen import CPG, Commissural
import networkx as nx


def main():
    """Main."""

    #: CPG
    net1 = CPG('FL', anchor_x=0., anchor_y=0.)  #: Directed graph
    net2 = CPG('FR', anchor_x=8., anchor_y=0.)  #: Directed graph
    net3 = CPG('HL', anchor_x=0., anchor_y=8.)  #: Directed graph
    net4 = CPG('HR', anchor_x=8., anchor_y=8.)  #: Directed graph

    #: Commussiral
    net5 = Commissural('FL', anchor_x=3.5, anchor_y=0.,
                       color='c')  #: Directed graph
    net6 = Commissural('FR', anchor_x=4.5, anchor_y=0.,
                       color='c')  #: Directed graph
    net7 = Commissural('HL', anchor_x=3.5, anchor_y=8.,
                       color='c')  #: Directed graph
    net8 = Commissural('HR', anchor_x=4.5, anchor_y=8.,
                       color='c')  #: Directed graph

    #: Connecting sub graphs
    net = nx.compose_all([net1.cpg,
                          net2.cpg,
                          net3.cpg,
                          net4.cpg,
                          net5.commissural,
                          net6.commissural,
                          net7.commissural,
                          net8.commissural])

    #: Connect Nodes Between Sub-Networks

    net.add_edge('FL_RG_F', 'FL_CINe1', weight=0.25)
    net.add_edge('FL_RG_F', 'FL_CINe2', weight=0.65)
    net.add_edge('FL_RG_F', 'FL_CINi1', weight=0.4)
    net.add_edge('FL_RG_E', 'FL_CINi2', weight=0.3)

    net.add_edge('FR_RG_F', 'FR_CINe1', weight=0.25)
    net.add_edge('FR_RG_F', 'FR_CINe2', weight=0.65)
    net.add_edge('FR_RG_F', 'FR_CINi1', weight=0.4)
    net.add_edge('FR_RG_E', 'FR_CINi2', weight=0.3)

    net.add_edge('HL_RG_F', 'HL_CINe1', weight=0.25)
    net.add_edge('HL_RG_F', 'HL_CINe2', weight=0.65)
    net.add_edge('HL_RG_F', 'HL_CINi1', weight=0.4)
    net.add_edge('HL_RG_E', 'HL_CINi2', weight=0.3)

    net.add_edge('HR_RG_F', 'HR_CINe1', weight=0.25)
    net.add_edge('HR_RG_F', 'HR_CINe2', weight=0.65)
    net.add_edge('HR_RG_F', 'HR_CINi1', weight=0.4)
    net.add_edge('HR_RG_E', 'HR_CINi2', weight=0.3)

    net.add_edge('FL_Ini1', 'FL_RG_F', weight=-0.2)
    net.add_edge('FR_CINi1', 'FL_RG_F', weight=-0.0266)
    net.add_edge('FR_CINi2', 'FL_RG_F', weight=-0.012)
    net.add_edge('FR_CINe1', 'FL_RG_F', weight=0.02)

    net.add_edge('FR_Ini1', 'FR_RG_F', weight=-0.2)
    net.add_edge('FL_CINi1', 'FR_RG_F', weight=-0.0266)
    net.add_edge('FL_CINi2', 'FR_RG_F', weight=-0.012)
    net.add_edge('FL_CINe1', 'FR_RG_F', weight=0.02)

    net.add_edge('HL_Ini1', 'HL_RG_F', weight=-0.3)
    net.add_edge('HR_CINi1', 'HL_RG_F', weight=-0.04)
    net.add_edge('HR_CINi2', 'HL_RG_F', weight=-0.017)
    net.add_edge('HR_CINe1', 'HL_RG_F', weight=0.03)

    net.add_edge('HR_Ini1', 'HR_RG_F', weight=-0.3)
    net.add_edge('HL_CINi1', 'HR_RG_F', weight=-0.04)
    net.add_edge('HL_CINi2', 'HR_RG_F', weight=-0.017)
    net.add_edge('HL_CINe1', 'HR_RG_F', weight=0.03)

    net.add_edge('FR_CINe2', 'FL_Ini1', weight=0.35)
    net.add_edge('FL_CINe2', 'FR_Ini1', weight=0.35)
    net.add_edge('HR_CINe2', 'HL_Ini1', weight=0.35)
    net.add_edge('HL_CINe2', 'HR_Ini1', weight=0.35)

    nx.write_graphml(net,
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
