""" Anne CPG Model. """

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import biolog
from anne_net_gen import CPG
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
    cpg = CPG('main', 30.0, 0.0)

    net = nx.compose_all([cpg.cpg])

    #: Location to save the network
    net_dir = os.path.join(
        os.path.dirname(__file__),
        './conf/auto_gen_anne_cpg.graphml')
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
    net_ = NetworkGenerator('./conf/auto_gen_anne_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 0.01  #: Time step
    time_vec = np.arange(0, 3, dt)  #: Time
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
        phase = cpg.rhythm_generator(time_vec[idx], 1, human_sys = None)
        if phase < 0.62:
            net_.neurons['main_Right_Flexor'].ext_in.val = 0.6
        else:
            net_.neurons['main_Right_Flexor'].ext_in.val = 0.
            
        if phase > 0.38:
            net_.neurons['main_Left_Flexor'].ext_in.val = 0.6
        else:
            net_.neurons['main_Left_Flexor'].ext_in.val = 0.
            
        res[idx] = net_.step()['xf'].full()[:, 0]
    end_time = time.time()

    biolog.info('Execution Time : {}'.format(
        end_time - start_time))

    # #: Results
    # net_.save_network_to_dot()
    ##: Visualize network using Matplotlib
    #fig = net_.visualize_network(plt_out=plt)
    plt.figure()
    plt.plot(time_vec,res[:,0])
    plt.plot(time_vec,res[:,2])
    #plt.plot(time_vec,res[:,3])
    #plt.plot(time_vec,res[:,5])
    plt.xlabel('Time [s]')
    plt.ylabel('CPG activation signal')
    plt.legend(['Left Extensor', 'Left Flexor'])#, 'Right Extensor', 'Right Flexor'])
    print(res)
    plt.show()
    

if __name__ == '__main__':
    main()
