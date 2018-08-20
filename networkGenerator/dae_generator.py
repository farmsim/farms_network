""" Describe/Generate DAE Model."""

from collections import OrderedDict
import casadi as cas
import biolog

class Parameters(object):
    """Book keeping for casadi objects.
    """

    def __init__(self):
        """ Initialization. """
        super(Parameters, self).__init__()
        self.data = OrderedDict()  #: Ordered Dictionary
        return

    def add_data(self, name, value):
        """Add data to the list.
        Parameters
        ----------
        name : <str>
            Name of the value to be saved as
        value : <float> <cas.SX.sym> <cas.SX.sym>
            Data to be stored. Can be of any type
        """
        if name not in self.data:
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


class DaeGenerator(object):
    """Dae Generator class

    """
    def __init__(self):
        """ Initialization. """
        super(DaeGenerator, self).__init__()

        #: Variables
        self.x = Parameters()  #: States
        self.z = Parameters()  #: Algebraic variables
        self.p = Parameters()  #: Parameters
        self.u = Parameters()  #: Inputs
        self.c = Parameters()  #: Named Constants
        self.y = Parameters()  #: Outputs

        #: Equations
        self.alg = Parameters()  #: Algebraic equations
        self.ode = Parameters()  #: ODE RHS
        self.dae = {}  #: DAE

    def add_x(self, name):
        """Add a new state.
        Parameters
        ----------
        name : <str>
            Name of the new state to be defined
        """
        state = cas.SX.sym(name)
        self.x.add_data(name, state)
        return state

    def add_z(self, name):
        """Add a new algebraic variable.
        Parameters
        ----------
        name : <str>
            Name of the new algebraic variable to be defined
        """
        alg_var = cas.SX.sym(name)
        self.z.add_data(name, alg_var)
        return alg_var

    def add_p(self, name):
        """Add a new parameter.
        Parameters
        ----------
        name : <str>
            Name of the new param to be defined
        """
        param = cas.SX.sym(name)
        self.p.add_data(name, param)
        return param

    def add_u(self, name):
        """Add a new Input.
        Parameters
        ----------
        name : <str>
            Name of the new input to be defined
        """
        inp = cas.SX.sym(name)
        self.u.add_data(name, inp)
        return inp

    def add_c(self, name):
        """Add a new constant.
        Parameters
        ----------
        name : <str>
            Name of the new constant to be defined
        """
        const = cas.SX.sym(name)
        self.c.add_data(name, const)
        return const

    def add_y(self, name):
        """Add a new output.
        Parameters
        ----------
        name : <str>
            Name of the new output to be defined
        """
        out = cas.SX.sym(name)
        self.y.add_data(name, out)
        return out

    def add_ode(self, name, ode):
        """Add a new ode rhs.
        Parameters
        ----------
        name : <str>
            Name of the new ode
        ode : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.ode.add_data(name, ode)
        return

    def add_alg(self, name, alg_eqn):
        """Add a new algebraic equation.
        Parameters
        ----------
        name : <str>
            Name of the new ode
        alg_eqn : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.alg.add_data(name, alg_eqn)
        return

    def generate_dae(self):
        """Generate DAE for the full network.
        Returns
        ----------
        dae : <dict>
            Differential algebraic equation of the network
        """
        #: For variable time step pylint: disable=invalid-name
        self.dae = {'x': self.x.to_list(),
                    'p': self.u.to_list() + self.p.to_list() + self.c.to_list(),
                    'z': self.z.to_list(),
                    'alg': self.alg.to_list(),
                    'ode': self.ode.to_list()}
        return self.dae
