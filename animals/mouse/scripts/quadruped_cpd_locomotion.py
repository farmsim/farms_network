""" Quadruped cpg locomotion controller. """

import farms_pylog as pylog
import networkx as nx
import os
from farms_network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container
from farms_sdf.sdf import ModelSDF
