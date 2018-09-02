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
        self.param_list = []  #: List to store param objects
        self._name_to_idx = {}  #: Dict to store name, index
        return

    def get_idx(self, key):
        """ Get the attribute value by name."""
        if key not in self._name_to_idx:
            raise AttributeError()
        else:
            return self.param_list[self._name_to_idx[key]].idx

    def get_val(self, key):
        """ Get the attribute value by name."""
        if key not in self._name_to_idx:
            raise AttributeError()
        else:
            return self[self._name_to_idx[key]]

    def set_val(self, key, value):
        """ Set the attribute value by name."""
        if key not in self._name_to_idx:
            raise AttributeError()
        else:
            self[self._name_to_idx[key]] = value

    def set_all_val(self, value):
        """ Set the attribute value by name."""
        for idx, _ in enumerate(self):
            self[idx] = value

    def get_sym(self, key):
        """ Get the attribute symbol by name."""
        if key not in self._name_to_idx:
            raise AttributeError()
        else:
            return self.param_list[self._name_to_idx[key]].sym

    def set_sym(self, key, sym):
        """ Get the attribute symbol by name."""
        if key not in self._name_to_idx:
            raise AttributeError()
        else:
            biolog.warning('Overwriting existing symbol')
            self.param_list[self._name_to_idx[key]].sym = sym

    def get_all_sym(self):
        """ Get list of all symbols. """
        return [param.sym for param in self.param_list]

    def add(self, name, sym, value=None):
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
        _param = Param(p_list=self, idx=_idx, sym=sym, value=value)
        self.param_list.insert(_idx, _param)

        return


class Param(object):
    """Wrapper for parameters in the system.
    Can be a casadi symbolic or numeric value.
    """

    def __init__(self, p_list, idx, sym, value):
        """ Initialization. """
        super(Param, self).__init__()

        #: List
        self._p_list = p_list

        #: idx
        self.idx = idx

        #:
        self._p_list.insert(self.idx, value)

        #: Update the value to the list
        self.val = value

        #: Store the symbol
        self.sym = sym

        return

    @property
    def val(self):
        """ Set the value of the parameter  """
        return self._p_list[self.idx]

    @val.setter
    def val(self, data):
        """
        Set the value of the parameter
        Parameters
        ----------
        data : <float>
            Value of the parameter
        """
        try:
            self._p_list[self.idx] = data
        except IndexError:
            biolog.error('Unable to find index {} in list {}'.format(
                self.idx, self._p_list))
            raise IndexError()


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
         ----------
        name : <str>
            Name of the new state to be defined
        value : <float>
            Initial value for the state variable
        """

        self.x.add(name, cas.SX.sym(name), value)
        return self.x.param_list[-1]

    def add_z(self, name, value=0.0):
        """Add a new algebraic variable.
         ----------
        name : <str>
            Name of the new algebraic variable to be defined
        value : <float>
            Initial value for the algebraic variable
        """
        self.z.add(name, cas.SX.sym(name), value)
        return self.z.param_list[-1]

    def add_p(self, name, value=0.0):
        """Add a new parameter.
         ----------
        name : <str>
            Name of the new param to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        self.p.add(name, cas.SX.sym(name), value)
        return self.p.param_list[-1]

    def add_u(self, name, value=0.0):
        """Add a new Input.
         ----------
        name : <str>
            Name of the new input to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        self.u.add(name, cas.SX.sym(name), value)
        return self.u.param_list[-1]

    def add_c(self, name, value=0.0):
        """Add a new constant.
         ----------
        name : <str>
            Name of the new constant to be defined
        value : <float>
            Default parameter value to be used during run time
        """
        self.c.add(name, cas.SX.sym(name), value)
        return self.c.param_list[-1]

    def add_ode(self, name, ode):
        """Add a new ode rhs.
         ----------
        name : <str>
            Name of the new ode
        ode : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.ode.add(name, ode)
        return self.ode.param_list[-1]

    def add_alg(self, name, alg_eqn):
        """Add a new algebraic equation.
         ----------
        name : <str>
            Name of the new ode
        alg_eqn : <cas.SX.sym>
            Symbolic ODE equation
        """
        self.alg.add(name, alg_eqn)
        return self.alg.param_list[-1]

    def generate_dae(self):
        """Generate DAE for the full network.
        Returns
        ----------
        dae : <dict>
            Differential algebraic equation of the network
        """
        #: For variable time step pylint: disable=invalid-name
        dae = {'x': cas.vertcat(*self.x.get_all_sym()),
               'p': cas.vertcat(*self.u.get_all_sym() +
                                self.p.get_all_sym() +
                                self.c.get_all_sym()),
               'z': cas.vertcat(*self.z.get_all_sym()),
               'alg': cas.vertcat(*self.alg.get_all_sym()),
               'ode': cas.vertcat(*self.ode.get_all_sym())}
        return dae

    def print_dae(self):
        """ Pretty print. """

        biolog.info(15 * '#' +
                    ' STATES : {} '.format(len(self.x)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.x.get_all_sym())]))

        biolog.info(15 * '#' +
                    ' INPUTS : {} '.format(len(self.u)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.u.get_all_sym())]))

        biolog.info(15 * '#' +
                    ' PARAMETERS : {} '.format(len(self.p)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.p.get_all_sym())]))

        biolog.info(15 * '#' +
                    ' CONSTANTS : {} '.format(len(self.c)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.c.get_all_sym())]))

        biolog.info(15 * '#' +
                    ' ODE : {} '.format(len(self.ode)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.ode.get_all_sym())]))

        biolog.info(15 * '#' +
                    ' ALGEBRAIC VARIABLES : {} '.format(len(self.z)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.z.get_all_sym())]))

        biolog.info(15 * '#' +
                    ' ALGEBRAIC EQUATIONS : {} '.format(len(self.alg)) +
                    15 * '#')
        print('\n'.join(['{}. {}'.format(
            j, s) for j, s in enumerate(self.alg.get_all_sym())]))
        return
