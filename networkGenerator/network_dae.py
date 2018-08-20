""" File contains the class to encapsulate network dae model."""

import casadi as cas
import biolog
from collections import OrderedDict


class Parameters(object):
    """Book keeping of casadi objects.
    """

    def __init__(self):
        """ Initialize. """
        super(Parameters, self).__init__()
        self.data = OrderedDict()  #: Ordered data

    def add_data(self, name, value):
        """Add data to the list.
        Parameters
        ----------
        name : <str>
            Name of the value to be saved as
        value : <float> <cas.SX.sym> <cas.SX.sym>
            Data to be stored. Can be of any type
        """
        if not self.data.has_key(name):
            self.data[name] = value
        else:
            biolog.error(
                'Can not add value {}  with name : {}'.format(
                    value, name))
            raise AttributeError()
        return

    def get_data(self, name):
        """ Return the appropriate data.
        Parameters
        ----------
        self: type
            description
        name: <str>
            Name of the data to be queried
        Returns
        -------
        data: <cas.SX.sym>
            Data attributed to the give name
        """
        try:
            return self.data[name]
        except KeyError:
            biolog.error('Undefined key {}'.format(name))
            raise KeyError()

    def to_list(self):
        """ Return the data as a list."""
        return [val for val in self.data.values()]


class NetworkDae(object):
    """Class encapsulating the Neural Network Dae.
    """

    def __init__(self):
        """ Initialization."""
        super(NetworkDae, self).__init__()

        self.states = Parameters()  #: States
        self.params = Parameters()  #: Parameters
        self.ode = Parameters()  #: ODE equations
        self.alg_vars = Parameters()  #: Algebraic variables
        self.alg_eqns = Parameters()  #: Algebraic equations

        self.dae = {}  #: DAE for integration

    def add_state(self, name):
        """Add a new state to the network.
        Parameters
        ----------
        name : <str>
            Name of the new state to be defined
        """
        state = cas.SX.sym(name)
        self.states.add_data(name, state)
        return state

    def add_param(self, name):
        """Add a new parameter to the network.
        Parameters
        ----------
        name : <str>
            Name of the new param to be defined
        """
        param = cas.SX.sym(name)
        self.params.add_data(name, param)
        return param

    def add_ode(self, name, ode):
        """Add a new ode rhs to the network.
        Parameters
        ----------
        name : <str>
            Name of the new ode
        ode : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.ode.add_data(name, ode)
        return

    def add_alg_var(self, name):
        """Add a new algebraic variable to the network.
        Parameters
        ----------
        name : <str>
            Name of the new algebraic variable
        """
        alg_var = cas.SX.sym(name)
        self.alg_vars.add_data(name, alg_var)
        return alg_var

    def add_alg_eqn(self, name, alg_eqn):
        """Add a new algebraic equation to the network.
        Parameters
        ----------
        name : <str>
            Name of the new ode
        alg_eqn : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.alg_eqns.add_data(name, alg_eqn)
        return

    def generate_dae(self):
        """Generate DAE for the full network.
        Returns
        ----------
        dae : <dict>
            Differential algebraic equation of the network
        """
        #: For variable time step pylint: disable=invalid-name
        self.dae = {'x': self.states.to_list(),
                    'p': self.params.to_list(),
                    'z': self.alg_vars.to_list(),
                    'alg': self.alg_eqns.to_list(),
                    'ode': self.ode.to_list()}
        return self.dae
