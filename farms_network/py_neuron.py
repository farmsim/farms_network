"""Abstract class for writing neuron models"""

# SIX to be removed after python2 support is dropped

import abc
import six
import farms_pylog as biolog
biolog.set_level('error')


@six.add_metaclass(abc.ABCMeta)
class Neuron():
    """Base neuron class.
    Inherits from Casadi Dae Builder Class.
    """

    def __init__(self, neuron_type):
        super(Neuron, self).__init__()
        self._neuron_type = neuron_type  # : Type of neuron

    @property
    def neuron_type(self):
        """Neuron type.  """
        return self._neuron_type

    @neuron_type.setter
    def neuron_type(self, value):
        """
        Parameters
        ----------
        value : <str>
            Type of neuron model
        """
        self._neuron_type = value

    @abc.abstractmethod
    def add_ode_input(self, neuron, **kwargs):
        """Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        kwargs : <dict>
             Contains the weight/synaptic information from the receiving neuron.
        """
        biolog.error('add_ode_input : Method not implemented in child class')
        raise NotImplementedError()

    @abc.abstractmethod
    def ode_rhs(self):
        """ ODE RHS.
        Returns
        ----------
        ode_rhs: <list>
            List containing the rhs equations of the ode states in the system
        """
        biolog.error('ode_rhs : Method not implemented in child class')
        raise NotImplementedError()

    def ode_alg_eqn(self):
        """ ODE Algebraic equations.
        Returns
        ----------
        alg_eqn: <list>
            List containing the ode algebraic equations
        """
        biolog.error('ode_alg_eqn : Method not implemented in child class')
        raise NotImplementedError()

    @abc.abstractmethod
    def neuron_out(self):
        """ Output of the neuron model.
        Returns
        ----------
        out: <cas.SX.sym>
            Output of the neuron  model
        """
        biolog.error('neuron_out : Method not implemented in child class')
        raise NotImplementedError()
