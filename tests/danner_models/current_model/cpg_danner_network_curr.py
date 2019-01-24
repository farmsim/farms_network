""" Danner CPG Model. Current Model """

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.matlib as npml
from matplotlib import rc
import os
from scipy.stats import circstd, circmean


import biolog
from danner_net_gen_curr import (CPG, LPSN, Commissural, ConnectFore2Hind,
                                   ConnectRG2Commissural,PatternFormation,
                                   ConnectPF2RG,Motorneurons,
                                   ConnectMN2CPG)
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
    #: CPG
    net_cpg1 = CPG('HL', anchor_x=0., anchor_y=40.)  #: Directed graph
    net_cpg2 = CPG('HR', anchor_x=40., anchor_y=40.)  #: Directed graph
    net_cpg3 = CPG('FL', anchor_x=0., anchor_y=-20.)  #: Directed graph
    net_cpg4 = CPG('FR', anchor_x=40., anchor_y=-20.)  #: Directed graph

    #: PATTERN FORMATION LAYER
    net_pf1 = PatternFormation('HL', anchor_x=0.,
                               anchor_y=45.)  #: Directed graph
    net_pf2 = PatternFormation(
        'HR', anchor_x=40., anchor_y=45.)  #: Directed graph
    net_pf3 = PatternFormation('FL', anchor_x=0.,
                               anchor_y=-15.)  #: Directed graph
    net_pf4 = PatternFormation(
        'FR', anchor_x=40., anchor_y=-15.)  #: Directed graph

    #: Commussiral
    net_comm1 = Commissural('HL', anchor_x=10., anchor_y=20.)
    net_comm2 = Commissural('HR', anchor_x=30., anchor_y=20.)
    net_comm3 = Commissural('FL', anchor_x=10., anchor_y=10.)
    net_comm4 = Commissural('FR', anchor_x=30., anchor_y=10.)

    #: Ipsilateral
    net9 = LPSN('L', anchor_x=15., anchor_y=20.,
                color='c')  #: Directed graph
    net10 = LPSN('R', anchor_x=25., anchor_y=20.,
                 color='c')  #: Directed graph

    #: Connecting sub graphs
    net_rg_pf1 = ConnectPF2RG(net_cpg1.cpg, net_pf1.pf_net)
    net_rg_pf2 = ConnectPF2RG(net_cpg2.cpg, net_pf2.pf_net)
    net_rg_pf3 = ConnectPF2RG(net_cpg3.cpg, net_pf3.pf_net)
    net_rg_pf4 = ConnectPF2RG(net_cpg4.cpg, net_pf4.pf_net)
    
    #: Motorneurons
    hind_muscles = ['PMA', 'CF', 'SM', 'POP', 'RF', 'TA', 'SOL', 'LG']
    net_motorneurons_hl = Motorneurons('HL', hind_muscles, anchor_x=0.,
                                       anchor_y=60.)
    net_motorneurons_hr = Motorneurons('HR', hind_muscles, anchor_x=40.,
                                       anchor_y=60.)
    fore_muscles = ['HFL','HEX','KFL','KEX','AFL','AEX']
    net_motorneurons_fl = Motorneurons('FL', hind_muscles, anchor_x=0.,
                                       anchor_y=-60.)
    net_motorneurons_fr = Motorneurons('FR', hind_muscles, anchor_x=40.,
                                       anchor_y=-60.)

    net_rg_pf_mn1 = ConnectMN2CPG(net_rg_pf1.net, net_motorneurons_hl.net)
    net_rg_pf_mn2 = ConnectMN2CPG(net_rg_pf2.net, net_motorneurons_hr.net)
    net_rg_pf_mn3 = ConnectMN2CPG(net_rg_pf3.net, net_motorneurons_fl.net)
    net_rg_pf_mn4 = ConnectMN2CPG(net_rg_pf4.net, net_motorneurons_fr.net) 
    
    net_RG_CIN1 = ConnectRG2Commissural(rg_l=net_rg_pf_mn1.net, rg_r=net_rg_pf_mn2.net,
                                        comm_l=net_comm1.commissural,
                                        comm_r=net_comm2.commissural)
    net_RG_CIN2 = ConnectRG2Commissural(rg_l=net_rg_pf_mn3.net, rg_r=net_rg_pf_mn4.net,
                                        comm_l=net_comm3.commissural,
                                        comm_r=net_comm4.commissural)

    net = ConnectFore2Hind(net_RG_CIN1.net,
                           net_RG_CIN2.net, net9.lpsn,
                           net10.lpsn)

    net = net.net
    
    #: Location to save the network
    net_dir = os.path.join(
        os.path.dirname(__file__),
        './conf/auto_gen_danner_current.graphml')
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
    net_ = NetworkGenerator(os.path.join(os.path.dirname(__file__),'./conf/auto_gen_danner_current.graphml'))

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    dur = 6000
    time_vec = np.arange(0, dur, dt)  #: Time

    #: Vector to store results
    res = np.empty([len(time_vec), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            "nonlinear_solver_iteration": "functional",
            "linear_multistep_method":"bdf",
            'jit': True,
            "enable_jacobian": True,
            "print_time": False,
            "print_stats": False,
            "reltol": 1e-4,
            "abstol": 1e-3}
    

    # #: Setup the integrator
    net_.setup_integrator(integration_method='cvodes',
                          opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')

    #: Network drive : Alpha
    alpha = np.linspace(0, 1, len(time_vec))

    start_time = time.time()
    for idx, _ in enumerate(time_vec):
        net_.dae.u.set_all_val(alpha[idx])
        res[idx] = net_.step()['xf'].full()[:, 0]
    end_time = time.time()

    biolog.info('Execution Time : {}'.format(
        end_time - start_time))


    # #: Results
    def get_gait_plot_from_neuron_act(act):
        """ Get start and end times of neurons for gait plot. """
        act_binary = (np.array(act) > 0.1).astype(np.int)
        act_binary = np.logical_not(act_binary).astype(np.int)
        act_binary[0] = 0
        gait_cycle = []
        start = (np.where(np.diff(act_binary[:, 0]) == 1.))[0]
        end = (np.where(np.diff(act_binary[:, 0]) == -1.))[0]
        for id, val in enumerate(start[:len(end)]):
            #: HARD CODED TIME SCALING HERE!!
            gait_cycle.append((val*0.001, end[id]*0.001 - val*0.001))
        return gait_cycle

    net_.save_network_to_dot()
    net_.visualize_network(node_size=250,
                           node_labels=False,
                           edge_labels=False,
                           edge_alpha=False,
                           plt_out=plt)  #: Visualize network using Matplotlib


    plot_names = ['FR_RG_F','FL_RG_F','HR_RG_F','HL_RG_F']
    

    plot_names = ['FR_RG_F','FL_RG_F','HR_RG_F','HL_RG_F',
                    'HL_RG_E','HL_PF_F','HL_PF_E','HL_PF_Sw','HL_PF_St',
                    'HL_Mn_PMA','HL_Mn_CF','HL_Mn_SM']
    plot_traces = list()
    for n in plot_names: 
        plot_traces.append(net_.neurons[n].neuron_out(
                    res[:, net_.dae.x.get_idx('V_'+n)]))

    fig, ax = plt.subplots(len(plot_names)+2, 1, sharex='all')
    fig.canvas.set_window_title('Model Performance')
    fig.suptitle('Model Performance', fontsize=12)
    for i,tr in enumerate(plot_traces):
        ax[i].plot(time_vec*0.001, tr, 'b',
                linewidth=1)
        ax[i].grid('on', axis='x')
        ax[i].set_ylabel(plot_names[i],fontsize=10)
        ax[i].set_yticks([0, 1])


    _width = 0.2
    colors = ['blue','green','red','black']
    for i,tr in enumerate(plot_traces):
        if i>3:
            break
        ax[len(plot_names)].broken_barh(get_gait_plot_from_neuron_act(tr),
                    (1.6-i*0.2, _width), facecolors=colors[i])
    ax[len(plot_names)].broken_barh(get_gait_plot_from_neuron_act(plot_traces[3]),
                    (1.0, _width*4), facecolors=(0.2, 0.2, 0.2), alpha=0.5)
    ax[len(plot_names)].set_ylim(1.0, 1.8)
    ax[len(plot_names)].set_xlim(0)
    ax[len(plot_names)].set_xlabel('Time')
    ax[len(plot_names)].set_yticks([1.1, 1.3, 1.5, 1.7])
    ax[len(plot_names)].set_yticklabels(['HL', 'HR', 'FL', 'FR'])
    ax[len(plot_names)].grid(True)

    ax[len(plot_names)+1].fill_between(time_vec*0.001, 0, alpha,
                     color=(0.2, 0.2, 0.2), alpha=0.5)
    ax[len(plot_names)+1].grid('on', axis='x')
    ax[len(plot_names)+1].set_ylabel('ALPHA')
    ax[len(plot_names)+1].set_xlabel('Time [s]')
    
    plt.show()


if __name__ == '__main__':
    main()
