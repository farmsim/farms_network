""" Danner CPG Model. """

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from network_generator.network_generator import NetworkGenerator

import biolog
from danner_net_gen import CPG, Commissural, Ipsilateral

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

    #: Ipsilateral
    net9 = Ipsilateral('L', anchor_x=0.0, anchor_y=5.,
                       color='c')  #: Directed graph

    net10 = Ipsilateral('R', anchor_x=10.0, anchor_y=5.,
                        color='c')  #: Directed graph
    #: Connecting sub graphs
    net = nx.compose_all([net1.cpg,
                          net2.cpg,
                          net3.cpg,
                          net4.cpg,
                          net5.commissural,
                          net6.commissural,
                          net7.commissural,
                          net8.commissural,
                          net9.ipsilateral,
                          net10.ipsilateral])

    #: Connect Nodes Between Sub-Networks

    net.add_edge('FL_RG_F', 'FL_CINe1', weight=0.25)
    net.add_edge('FL_RG_F', 'FL_CINe2', weight=0.65)
    net.add_edge('FL_RG_F', 'FL_CINi1', weight=0.4)
    net.add_edge('FL_RG_F', 'LF_Ini_fh', weight=0.5)
    net.add_edge('FL_RG_E', 'FL_CINi2', weight=0.3)

    net.add_edge('FR_RG_F', 'FR_CINe1', weight=0.25)
    net.add_edge('FR_RG_F', 'FR_CINe2', weight=0.65)
    net.add_edge('FR_RG_F', 'FR_CINi1', weight=0.4)
    net.add_edge('FR_RG_F', 'RF_Ini_fh', weight=0.5)
    net.add_edge('FR_RG_E', 'FR_CINi2', weight=0.3)

    net.add_edge('HL_RG_F', 'HL_CINe1', weight=0.25)
    net.add_edge('HL_RG_F', 'HL_CINe2', weight=0.65)
    net.add_edge('HL_RG_F', 'HL_CINi1', weight=0.4)
    net.add_edge('HL_RG_F', 'LH_Ini_fh', weight=0.5)
    net.add_edge('HL_RG_E', 'HL_CINi2', weight=0.3)

    net.add_edge('HR_RG_F', 'HR_CINe1', weight=0.25)
    net.add_edge('HR_RG_F', 'HR_CINe2', weight=0.65)
    net.add_edge('HR_RG_F', 'HR_CINi1', weight=0.4)
    net.add_edge('HR_RG_F', 'RH_Ini_fh', weight=0.5)
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

    net.add_edge('LF_Ini_fh', 'HL_RG_F', weight=-0.015)
    net.add_edge('LH_Ini_fh', 'FL_RG_F', weight=-0.035)
    net.add_edge('RF_Ini_fh', 'HR_RG_F', weight=-0.015)
    net.add_edge('RH_Ini_fh', 'FR_RG_F', weight=-0.035)

    nx.write_graphml(net,
                     './conf/auto_gen_danner_cpg.graphml')

    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_danner_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time_vec = np.arange(0, 100000, dt)  #: Time
    #: Vector to store results
    res = np.empty([len(time_vec), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            'jit': True,
            "enable_jacobian": True,
            "print_time": False,
            "print_stats": False,
            "reltol": 1e-6,
            "abstol": 1e-6}

    # #: Setup the integrator
    net_.setup_integrator(integration_method='cvodes',
                          opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')

    #: Network drive : Alpha
    alpha = np.linspace(0, 1, len(time_vec))

    start_time = time.time()
    for idx, _ in enumerate(time_vec):
        net_.dae.u.set_all_val(alpha)
        res[idx] = net_.step()['xf'].full()[:, 0]
    end_time = time.time()

    biolog.info('Execution Time : {}'.format(
        end_time - start_time))

    # #: Results
    net_.save_network_to_dot()
    net_.visualize_network(plt)  #: Visualize network using Matplotlib

    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex='all')
    ax1.plot(time_vec*0.001,
             res[:, net_.dae.x.get_idx('V_FR_RG_F')], 'b')
    ax1.grid('on', axis='x')
    ax1.set_ylabel('FR')
    ax2.plot(time_vec*0.001,
             res[:, net_.dae.x.get_idx('V_FL_RG_F')], 'g')
    ax2.grid('on', axis='x')
    ax2.set_ylabel('FL')
    ax3.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HR_RG_F')],
             'r')
    ax3.grid('on', axis='x')
    ax3.set_ylabel('HR')
    ax4.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HL_RG_F')],
             'k')
    ax4.grid('on', axis='x')
    ax4.set_ylabel('HL')
    ax5.fill_between(time_vec*0.001, 0, alpha,
                     color=(0.2, 0.2, 0.2), alpha=0.5)
    ax5.grid('on', axis='x')
    ax5.set_ylabel('ALPHA')
    ax5.set_xlabel('Time [s]')

    plt.show()


if __name__ == '__main__':
    main()
