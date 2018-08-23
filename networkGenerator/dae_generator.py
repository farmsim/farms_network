""" Describe/Generate DAE Model. """

from collections import OrderedDict
import casadi as cas
import biolog


class Parameters(list):
    """Book keeping for casadi objects.
    """

    def __init__(self):
        """ Initialization. """
        super(Parameters, self).__init__()
        self._name_to_idx = {}  #: Dict to store name, index
        return

    def get(self, key):
        """ Get the attribute value by name."""
        if key not in self._name_to_idx:
            raise AttributeError()
        else:
            return self[self._name_to_idx[key]]

    def set(self, key, value):
        """ Set the attribute value by name."""
        if key not in self._name_to_idx:
            raise AttributeError()
        else:
            self[self._name_to_idx[key]] = value

    def add(self, name, sym, value):
        """Add new element to the parameter list.
        Parameters
        ----------
        name : <str>
             Name of the parameter

        sym : <cas.SX.sym>
             Symbolic casadi expression

        value : <float>
             Value to be set for the parameter

        Returns
        -------
        out : <bool>
            Returns true if successful
        """

        _idx = len(self)
        self._name_to_idx[name] = _idx
        self.insert(_idx, Param(sym=sym, value=value))
        return

class Param(object):
    """Wrapper for parameters in the system.
    Can be a casadi symbolic or numeric value.
    """

    def __init__(self, sym, value=None):
        """ Initialization. """
        super(Param, self).__init__()

        #: Update the value to the list
        self.val = value
        #: Store the symbol
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
        []       ----------
        name : <str>
            Name of the new state to be defined
        value : <float>
            Initial value for the state variable
        """

        self.x.append(Param(cas.SX.sym(name), value, self.x_l,
                            len(self.x_l)))
        return self.x[-1]

    def add_z(self, name, value=0.0):
        """Add a new algebraic variable.
        []       ----------
        name : <str>
            Name of the new algebraic variable to be defined
        value : <float>
            Initial value for the algebraic variable
        """
        self.z.append((cas.SX.sym(
            name), value, self.z_l, len(self.z_l)))
        return self.z[-1]

    def add_p(self, name, value=0.0):
        """Add a new parameter.
        []       ----------
        name : <str>
            Name of the new param to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        # self.p.append(Param(cas.SX.sym(
        #     name), value, self.p_l, len(self.p_l))
        # return self.p[-1]
        pass

    def add_u(self, name, value=0.0):
        """Add a new Input.
        []       ----------
        name : <str>
            Name of the new input to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        # self.u.append(Param(cas.SX.sym(
        #     name), value, self.u_l, len(self.u_l))
        # return self.u[-1]
        pass

    def add_c(self, name, value=0.0):
        """Add a new constant.
        []       ----------
        name : <str>
            Name of the new constant to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        # self.c.append(Param(cas.SX.sym(
        #     name), value, self.c_l, len(self.c_l))
        # return self.c[-1]
        pass

    def add_ode(self, name, ode):
        """Add a new ode rhs.
        []       ----------
        name : <str>
            Name of the new ode
        ode : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.ode[name]=Param(ode)
        return self.ode[name]

    def add_alg(self, name, alg_eqn):
        """Add a new algebraic equation.
        []       ----------
        name : <str>
            Name of the new ode
        alg_eqn : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.alg[name]=Param(alg_eqn)
        return self.alg[name]

    def generate_dae(self):
        """Generate DAE for the full network.
        Returns
        ----------
        dae : <dict>
            Differential algebraic equation of the network
        """
        #: For variable time step pylint: disable=invalid-name
        dae={'x': cas.vertcat(*self.x.syms()),
               'p': cas.vertcat(*self.u.syms() +
                                self.p.syms() + self.c.syms()),
               'z': cas.vertcat(*self.z.syms()),
               'alg': cas.vertcat(*self.alg.syms()),
               'ode': cas.vertcat(*self.ode.syms())}
        return dae

    def print_dae(self):
        """ Pretty print. """

        biolog.info(15 * '#' +
                    ' STATES : {} '.format(len(self.x)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.x.syms())]))

        biolog.info(15 * '#' +
                    ' INPUTS : {} '.format(len(self.u)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.u.syms())]))

        biolog.info(15 * '#' +
                    ' PARAMETERS : {} '.format(len(self.p)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.p.syms())]))

        biolog.info(15 * '#' +
                    ' CONSTANTS : {} '.format(len(self.c)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.c.syms())]))

        biolog.info(15 * '#' +
                    ' ODE : {} '.format(len(self.ode)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.ode.syms())]))

        biolog.info(15 * '#' +
                    ' ALGEBRAIC VARIABLES : {} '.format(len(self.z)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.z.syms())]))

        biolog.info(15 * '#' +
                    ' ALGEBRAIC EQUATIONS : {} '.format(len(self.alg)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.alg.syms())]))
        return
