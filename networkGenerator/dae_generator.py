""" Describe/Generate DAE Model."""

from collections import OrderedDict
import casadi as cas
import biolog


class Parameters(OrderedDict):
    """Book keeping for casadi objects.
    """

    def __init__(self):
        """ Initialization. """
        super(Parameters, self).__init__()
        return

    def vals(self):
        """ Get list parameter values. """
        return [val.val for val in self.values()]

    def syms(self):
        """ Get list parameter symbolics. """
        return [val.sym for val in self.values()]


class Param(object):
    """Wrapper for parameters in the system.
    Can be a casadi symbolic or numeric value.
    """

    def __init__(self, sym, val=None):
        """ Initialization. """
        super(Param, self).__init__()
        if val is not None:
            self.val = val
        self.sym = sym
        return


class DaeGenerator(object):
    """Dae Generator class
    """

    def __init__(self):
        """ Initialization. """
        super(DaeGenerator, self).__init__()

        #: Variables
        #: pylint: disable=invalid-name
        self.x = Parameters()  #: States
        self.z = Parameters()  #: Algebraic variables
        self.p = Parameters()  #: Parameters
        self.u = Parameters()  #: Inputs
        self.c = Parameters()  #: Named Constants
        self.y = Parameters()  #: Outputs

        #: Equations
        self.alg = Parameters()  #: Algebraic equations
        self.ode = Parameters()  #: ODE RHS

    def add_x(self, name, value=0.0):
        """Add a new state.
        Parameters
        ----------
        name : <str>
            Name of the new state to be defined
        value : <float>
            Initial value for the state variable
        """
        self.x[name] = Param(cas.SX.sym(name), value)
        return self.x[name]

    def add_z(self, name, value=0.0):
        """Add a new algebraic variable.
        Parameters
        ----------
        name : <str>
            Name of the new algebraic variable to be defined
        value : <float>
            Initial value for the algebraic variable
        """
        self.z[name] = Param(cas.SX.sym(name), value)
        return self.z[name]

    def add_p(self, name, value=0.0):
        """Add a new parameter.
        Parameters
        ----------
        name : <str>
            Name of the new param to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        self.p[name] = Param(cas.SX.sym(name), value)
        return self.p[name]

    def add_u(self, name, value=0.0):
        """Add a new Input.
        Parameters
        ----------
        name : <str>
            Name of the new input to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        self.u[name] = Param(cas.SX.sym(name), value)
        return self.u[name]

    def add_c(self, name, value=0.0):
        """Add a new constant.
        Parameters
        ----------
        name : <str>
            Name of the new constant to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        self.c[name] = Param(cas.SX.sym(name), value)
        return self.c[name]

    def add_y(self, name):
        """Add a new output.
        Parameters
        ----------
        name : <str>
            Name of the new output to be defined
        """
        self.y[name] = Param(cas.SX.sym(name), 0.0)
        return self.y[name]

    def add_ode(self, name, ode):
        """Add a new ode rhs.
        Parameters
        ----------
        name : <str>
            Name of the new ode
        ode : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.ode[name] = Param(ode)
        return self.ode[name]

    def add_alg(self, name, alg_eqn):
        """Add a new algebraic equation.
        Parameters
        ----------
        name : <str>
            Name of the new ode
        alg_eqn : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.alg[name] = Param(alg_eqn)
        return self.alg[name]

    def generate_dae(self):
        """Generate DAE for the full network.
        Returns
        ----------
        dae : <dict>
            Differential algebraic equation of the network
        """
        #: For variable time step pylint: disable=invalid-name
        dae = {'x': cas.vertcat(*self.x.syms()),
               'p': cas.vertcat(*self.u.syms() +
                                self.p.syms() + self.c.syms()),
               'z': cas.vertcat(*self.z.syms()),
               'alg': cas.vertcat(*self.alg.syms()),
               'ode': cas.vertcat(*self.ode.syms())}
        return dae
