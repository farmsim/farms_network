# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: optimize.unpack_method_calls=True
# cython: np_pythran=False

"""Sensory afferent neurons."""

from farms_network.neuron import Neuron
from libc.stdio cimport printf
from farms_container import Container

cdef class SensoryNeuron(Neuron):
    """Sensory afferent neurons connecting muscle model with the network.
    """

    def __init__(self, n_id, num_inputs, **kwargs):
        """Initialize.
        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(SensoryNeuron, self).__init__('sensory_neuron')

        #: Neuron ID
        self.n_id = n_id
        #: Get container
        container = Container.get_instance()

        self.aff_inp = container.neural.inputs.add_parameter(
            'aff_' + self.n_id, kwargs.get('init', 0.0))[0]

        #: Output
        self.nout = container.neural.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

    def reset_sensory_param(self, param):
        """ Add the sensory input. """
        self.aff_inp = param

    def add_ode_input(self, idx, neuron, **kwargs):
        """Abstract method"""
        pass


    def ode_rhs(self, y, p):
        """Abstract method"""
        self.c_ode_rhs(y, p)

    def output(self):
        """ Output of the neuron model.
        Returns
        ----------
        out: <cas.SX.sym>
            Output of the neuron  model
        """
        return self.c_output()

    #################### C-FUNCTIONS ####################

    cdef void c_ode_rhs(self, double[:] _y, double[:] _p) nogil:
        """ Compute the ODE. Internal Setup Function."""
        pass

    cdef void c_output(self) nogil:
        """ Neuron output. """
        #: Set the neuron output
        self.nout.c_set_value(self.aff_inp.c_get_value())
