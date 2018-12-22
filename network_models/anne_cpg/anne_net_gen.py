""" Generate Danner Network."""

import networkx as nx
import numpy as np


class CPG(object):
    """Generate CPG Network
    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0):
        """ Initialization. """
        super(CPG, self).__init__()
        self.cpg = nx.DiGraph()
        self.name = name
        self.cpg.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y):
        """ Add neurons. """
        self.cpg.add_node(self.name+'_Left_Extensor',
                          model='constant_and_inhibit',
                          x=-1.0+anchor_x,
                          y=0.0+anchor_y,
                          color='b',
                          K = -500,
                          B = -20,
                          ades = 0.4,
                          output=True)
                          
        self.cpg.add_node(self.name+'_Left_Flexor',
                          model='if',
                          x=1.0+anchor_x,
                          y=0.0+anchor_y,
                          color='b',
                          # standard constants
                          output=True)
        
        self.cpg.add_node(self.name+'_Right_Extensor',
                          model='constant_and_inhibit',
                          x=-1.0+anchor_x,
                          y=1.0+anchor_y,
                          color='b',
                          K = -500,
                          B = -20,
                          ades = 0.4,
                          output=True)
                          
        self.cpg.add_node(self.name+'_Right_Flexor',
                          model='if',
                          x=1.0+anchor_x,
                          y=1.0+anchor_y,
                          color='b',
                          # standard constants
                          output=True)
        
    def add_connections(self):
        self.cpg.add_edge(self.name+'_Left_Flexor',
                          self.name+'_Left_Extensor',
                          weight=1.,
                          K_f = 250)
        self.cpg.add_edge(self.name+'_Right_Flexor',
                          self.name+'_Right_Extensor',
                          weight=1.,
                          K_f = 250)
        return

    def rhythm_generator(self, time_elapsed, gait_duration, human_sys = None):
        #: Find the current time in the gait cycle
        phase = 1.*(time_elapsed % gait_duration)/gait_duration    
        
        #: Add some feedback: swing phase starts
        if human_sys is None:
            return phase
        else:
            if human_sys.swing['R'] == 1 and human_sys.ground_contacts['Ry'] > 0.1:
                phase = 0
            elif human_sys.swing['R'] == 1 and human_sys.ground_contacts['Ry'] > 0.1:
                phase = 0.38 #should this be  done differently
            
            #: Stance phase starts
            if human_sys.DS['R'] == 1 and human_sys.ground_contacts['Ry'] < 0.1:
                phase = 0.62
            elif human_sys.DS['R'] == 1 and human_sys.ground_contacts['Ry'] < 0.1:
                phase = 0 #should this be  done differently
                
            return phase
