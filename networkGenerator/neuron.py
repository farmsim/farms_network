"""Implementation of different neuron models."""
import biolog
import numpy as np


class Neuron(object):
    """Base neuron class.

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


class LeakyIntegrateFire(Neuron):
    """Leaky Integrate and Fire neuron.
    """

    def __init__(self, tau=0.2, D=1., bias=0.0):
        super(LeakyIntegrateFire, self).__init__(
            neuron_type='leaky_integrate_fire')
        self._tau = tau
        self._D = D
        self._bias = bias
        self.membrane = 0.0

    @property
    def tau(self):
        """Neuron time constant  """
        return self._tau

    @tau.setter
    def tau(self, value):
        """
        Parameters
        ----------
        value : <float>
            Time constant
        """
        self._tau = value

    @property
    def bias(self):
        """Neuron activation bias  """
        return self._bias

    @bias.setter
    def bias(self, value):
        """
        Parameters
        ----------
        value : <float>
            Neuron activation bias
        """
        self._bias = value

    @property
    def D(self):
        """Neuron D  """
        return self._D

    @D.setter
    def D(self, value):
        """
        Parameters
        ----------
        value : <float>
            Neuron D
        """
        self._D = value

    @property
    def membrane(self):
        """Neuron membrane potential  """
        return self._membrane

    @membrane.setter
    def membrane(self, value):
        """
        Parameters
        ----------
        value : <float>
            Neuron membrane potetional
        """
        self._membrane = value

    def ode(self, time, membrane, _in):
        """
        Parameters
        ----------
        _in : <np.1darray>
            Input currents to the neuron

        Returns
        -------
        out : <float>
            Rate of change of neuron membrane potential
            tau*dm\dt = -m + sum(_in)
        """
        self.membrane = membrane
        return (-self.membrane + np.sum(_in)) / self.tau

    def activation(self, _in):
        """
        Parameters
        ----------
        _in : <float>
            Membrane potential of the neuron

        Returns
        -------
        out : <float>
            Activation value of the neuron
        """
        _in = np.array(_in)
        return 1 / (1 + np.exp(-self.D * (_in + self.bias)))


def main():
    neuron1 = LeakyIntegrateFire()
    biolog.debug('Neuron type : {}'.format(neuron1.neuron_type))
    biolog.debug('Neuron ode out : {}'.format(neuron1.ode(np.arange(0, 5))))
    biolog.debug('Neuron activation out : {}'.format(
        neuron1.activation(neuron1.ode(np.arange(0, 5)))))


if __name__ == '__main__':
    main()
