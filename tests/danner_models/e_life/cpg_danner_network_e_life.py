""" Danner CPG Model. eLife """

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.matlib as npml
from matplotlib import rc
import os
from scipy.stats import circstd, circmean


import biolog
from danner_net_gen_e_life import (CPG, LPSN, Commissural, ConnectFore2Hind,
                                   ConnectRG2Commissural)
from network_generator.network_generator import NetworkGenerator

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def calc_phase(time_vec, out):
    out = np.piecewise(out, [out <= -50, out >= 0, np.logical_and(
        out > -50.0, out < 0.0)], [0, 1, lambda x:(x+50.0)/50.0])
    os_ = ((np.diff((out > 0.1).astype(np.int), axis=0) == 1).T)
    onsets = npml.repmat(time_vec[:-1], 4, 1)[os_]
    leg = (npml.repmat(np.arange(4.0), len(time_vec)-1, 1).T)[os_]
    times = np.stack((onsets, leg), 1)
    times = times[times[:, 0].argsort()]

    pdur = np.diff(times[(times[:, 1] == 0), 0])
    phases = np.zeros((3))
    std_phases = np.zeros((3))
    for i, (x, y) in enumerate([(0, 1), (0, 2), (0, 3)]):
        times_ = times[(times[:, 1] == x) | (times[:, 1] == y)]
        indices = np.where((times_[:-2, 1] == x) &
                           (times_[1:-1, 1] == y) & (times_[2:, 1] == x))
        pdur_ = times_[[ind+2 for ind in indices], 0]-times_[indices, 0]
        phases_ = ((times_[[ind+1 for ind in indices], 0] -
                    times_[indices, 0])/pdur_).T
        phases[i] = circmean(phases_[-6:-1], 1, 0)
        std_phases[i] = circstd(phases_[-6:-1], 1, 0)

    fq = 1/np.mean(pdur[-6:-1])
    return (fq, phases, np.max(std_phases))


def run_bifurcation(network, alphas, stepdur):
    dt = 1  #: Time step
    time_vec = np.arange(0, stepdur, dt)  #: Time

    frequencies = np.zeros((len(alphas)))*np.nan
    phases = np.zeros((len(alphas), 3))*np.nan

    inRGF_fr = network.dae.x.get_idx('V_FR_RG_F')
    inRGF_fl = network.dae.x.get_idx('V_FL_RG_F')
    inRGF_hr = network.dae.x.get_idx('V_HR_RG_F')
    inRGF_hl = network.dae.x.get_idx('V_HL_RG_F')
    # phase_diffs = [(inRGF_hl,inRGF_hr),(inRGF_hl,inRGF_fl),(inRGF_hl,inRGF_fr)]
    indices = (inRGF_hl, inRGF_hr, inRGF_fl, inRGF_fr)
    network.dae.u.set_all_val(alphas[0])
    for idx in range(1000):
        network.step()

    for i, alpha in enumerate(alphas):
        network.dae.u.set_all_val(alpha)
        res = np.empty([len(time_vec), len(network.dae.x)])
        for idx, _ in enumerate(time_vec):
            res[idx] = network.step()['xf'].full()[:, 0]

        fq_, phases_, stdp = calc_phase(time_vec*1e-3, res[:, indices])
        # import IPython; IPython.embed()
        frequencies[i] = fq_
        phases[i, :] = phases_
        biolog.info('Finished interation {}'.format(i))

    return (frequencies, phases)


def main():
    """Main."""

    #: CPG
    net1 = CPG('FL', anchor_x=-10., anchor_y=-10.)  #: Directed graph
    net2 = CPG('FR', anchor_x=10., anchor_y=-10.)  #: Directed graph
    net3 = CPG('HL', anchor_x=-10., anchor_y=15.)  #: Directed graph
    net4 = CPG('HR', anchor_x=10, anchor_y=15.)  #: Directed graph

    #: Commussiral
    net5 = Commissural('FL', anchor_x=-3, anchor_y=-10.,
                       color='c')  #: Directed graph
    net6 = Commissural('FR', anchor_x=3, anchor_y=-10.,
                       color='c')  #: Directed graph
    net7 = Commissural('HL', anchor_x=-3, anchor_y=15.,
                       color='c')  #: Directed graph
    net8 = Commissural('HR', anchor_x=3, anchor_y=15.,
                       color='c')  #: Directed graph

    #: Ipsilateral
    net9 = LPSN('L', anchor_x=-3., anchor_y=4.,
                color='c')  #: Directed graph
    net10 = LPSN('R', anchor_x=3., anchor_y=4.,
                 color='c')  #: Directed graph

    #: Connecting sub graphs

    net_RG_CIN1 = ConnectRG2Commissural(rg_l=net1.cpg, rg_r=net2.cpg,
                                        comm_l=net5.commissural,
                                        comm_r=net6.commissural)
    net_RG_CIN2 = ConnectRG2Commissural(rg_l=net3.cpg, rg_r=net4.cpg,
                                        comm_l=net7.commissural,
                                        comm_r=net8.commissural)

    net = ConnectFore2Hind(net_RG_CIN1.net,
                           net_RG_CIN2.net, net9.lpsn,
                           net10.lpsn)

    net = net.net

    #: Location to save the network
    net_dir = os.path.join(
        os.path.dirname(__file__),
        './conf/auto_gen_danner_cpg.graphml')
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
    net_ = NetworkGenerator(os.path.join(os.path.dirname(__file__),
                                         './conf/auto_gen_danner_cpg.graphml'))

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    dur = 6000
    time_vec = np.arange(0, dur, dt)  #: Time

    #: Vector to store results
    res = np.empty([len(time_vec), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            'jit': False,
            "enable_jacobian": True,
            "print_time": False,
            "print_stats": False,
            "reltol": 1e-4,
            "abstol": 1e-5}

    # #: Setup the integrator
    net_.setup_integrator(integration_method='cvodes',
                          opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')

    #: Network drive : Alpha
    alpha = np.floor(time_vec*1e-3)/(dur*1e-3-1.0)+0.05

    biolog.info('INPUTS')
    # print('\n'.join(['{} : {}'.format(p.sym.name(), p.val) for p in net_.dae.u.param_list]))

    start_time = time.time()
    net_.dae.u.set_all_val(alpha[0])
    for idx in range(1000):
        net_.step()

    for idx, _ in enumerate(time_vec):
        net_.dae.u.set_all_val(alpha[idx])
        res[idx] = net_.step()['xf'].full()[:, 0]
    end_time = time.time()

    biolog.info('Execution Time : {}'.format(
        end_time - start_time))

    alphas = np.linspace(0.02, 1.05, 100)
    fqs, phases = run_bifurcation(net_, alphas, 2000)

    alphas2 = np.linspace(1.05, 0.02, 100)
    fqs2, phases2 = run_bifurcation(net_, alphas2, 2000)

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='all')
    ax1.plot(alphas, fqs, 'b', linewidth=1)
    ax1.plot(alphas2, fqs2, 'r', linewidth=1)
    ax2.plot(alphas, phases[:, 0], 'b.')
    ax2.plot(alphas, 1.0-phases[:, 0], 'b.')
    ax2.plot(alphas2, phases2[:, 0], 'r.')
    ax2.plot(alphas2, 1.0-phases2[:, 0], 'r.')
    ax2.set_ylim([0.0, 1.0])
    ax3.plot(alphas, phases[:, 1], 'b.')
    ax3.plot(alphas2, phases2[:, 1], 'r.')
    ax3.set_ylim([0.0, 1.0])
    ax4.plot(alphas, phases[:, 2], 'b.')
    ax4.plot(alphas2, phases2[:, 2], 'r.')
    ax4.set_ylim([0.0, 1.0])

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
    net_.visualize_network(node_size=1250,
                           edge_labels=False,
                           plt_out=plt)  #: Visualize network using Matplotlib

    v_fr_rg_f = net_.neurons['FR_RG_F'].neuron_out(
        res[:, net_.dae.x.get_idx('V_FR_RG_F')])
    v_fl_rg_f = net_.neurons['FL_RG_F'].neuron_out(
        res[:, net_.dae.x.get_idx('V_FL_RG_F')])
    v_hr_rg_f = net_.neurons['HR_RG_F'].neuron_out(
        res[:, net_.dae.x.get_idx('V_HR_RG_F')])
    v_hl_rg_f = net_.neurons['HL_RG_F'].neuron_out(
        res[:, net_.dae.x.get_idx('V_HL_RG_F')])

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='all')
    fig.canvas.set_window_title('Model Performance')
    fig.suptitle('Model Performance', fontsize=16)
    ax1.plot(time_vec*0.001, v_fr_rg_f, 'b',
             linewidth=1)
    ax1.grid('on', axis='x')
    ax1.set_ylabel('FR')
    ax1.set_yticks([0, 1])
    ax2.plot(time_vec*0.001, v_fl_rg_f, 'g',
             linewidth=1)
    ax2.grid('on', axis='x')
    ax2.set_ylabel('FL')
    ax2.set_yticks([0, 1])
    ax3.plot(time_vec*0.001, v_hr_rg_f, 'r',
             linewidth=1)
    ax3.grid('on', axis='x')
    ax3.set_ylabel('HR')
    ax3.set_yticks([0, 1])
    ax4.plot(time_vec*0.001, v_hl_rg_f, 'k',
             linewidth=1)
    ax4.grid('on', axis='x')
    ax4.set_ylabel('HL')
    ax4.set_yticks([0, 1])

    _width = 0.2
    ax5.broken_barh(get_gait_plot_from_neuron_act(v_fr_rg_f),
                    (1.6, _width), facecolors='blue')
    ax5.broken_barh(get_gait_plot_from_neuron_act(v_fl_rg_f),
                    (1.4, _width), facecolors='green')
    ax5.broken_barh(get_gait_plot_from_neuron_act(v_hr_rg_f),
                    (1.2, _width), facecolors='red')
    ax5.broken_barh(get_gait_plot_from_neuron_act(v_hl_rg_f),
                    (1.0, _width), facecolors='black')
    ax5.broken_barh(get_gait_plot_from_neuron_act(v_hl_rg_f),
                    (1.0, _width*4), facecolors=(0.2, 0.2, 0.2), alpha=0.5)
    ax5.set_ylim(1.0, 1.8)
    ax5.set_xlim(0)
    ax5.set_xlabel('Time')
    ax5.set_yticks([1.1, 1.3, 1.5, 1.7])
    ax5.set_yticklabels(['HL', 'HR', 'FL', 'FR'])
    ax5.grid(True)

    ax6.fill_between(time_vec*0.001, 0, alpha,
                     color=(0.2, 0.2, 0.2), alpha=0.5)
    ax6.grid('on', axis='x')
    ax6.set_ylabel('ALPHA')
    ax6.set_xlabel('Time [s]')

    plt.show()


if __name__ == '__main__':
    main()
