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


class LIF_Interneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    """

    def __init__(self):
        super(LIF_Interneuron, self).__init__(
            neuron_type='lif_internueron')
        #: Neuron constants
        #: Parameters of INaP
        self.g_nap = 10.0
        self.e_nap = 50.0
        self.v_hm = -37.0
        self.gamma_m = -0.1667
        self.eps = 0.002

        #: Parameters of h
        self.v_hh = -30.0
        self.gamma_h = 0.1667
        self.v_htau = -30.0
        self.gamma_tau = 0.0833

        #: Parameters of Ileak
        self.g_leak = 2.8
        self.e_leak = -65.0

        #: Parameters of Isyn
        self.g_syn = 0.1
        self.e_syn = 0.0
        self.v_hs = -43.0
        self.gamma_s = -0.42

        #: Parameters of Iapp
        self.g_app = 0.16
        self.e_app = 0.0

        #: Other constants
        self.c_m = 1.0

    def ode(self, time, state, v_syn):
        """
        Parameters
        ----------
        time : <float>
            Current simulation time step
        states : <np.array>
            Neuron states
                * V : Membrane potential
                * h : Inactivation variable
        v_syn : <np.array>
            External synaptic membrane potentials
        Returns
        -------
        out : <np.array>
            Rate of change of neuron membrane potential

        """
        _V = state[0]  #: Membrane potential
        _h = state[1]  #: Inactivation variable

        def v_func(gam, v_h, v_ext=None):
            if v_ext is None:
                return 1. / (1 + np.exp(gam*(_V - v_h)))
            else:
                return 1. / (1 + np.exp(gam*(v_ext - v_h)))

        #: Inap
        I_nap = self.g_nap*v_func(
            self.gamma_m, self.v_hm)*_h*(_V - self.e_nap)

        #: Ileak
        I_leak = self.g_leak*(_V - self.e_leak)

        #: Iapp
        I_app = self.g_app*(_V - self.e_app)

        #: Isyn
        I_syn = np.sum(self.g_syn*v_func(
            self.gamma_s, self.v_hs, v_syn)*(_V - self.e_syn))

        #: hinf
        h_inf = v_func(self.gamma_h, self.v_hh)

        #: tau_h
        tau_h = 1. / (self.eps*(np.cosh(self.gamma_tau*(_V - self.v_htau))))

        #: dV
        _dV = -(I_nap + I_leak + I_syn + I_app)/self.c_m

        #: dh
        _dh = (h_inf - _h)/tau_h

        return np.array([_dV, _dh])


class LIF_Motorneuron(Neuron):
    """Leaky Integrate and Fire Motorneuron.
    """

    def __init__(self):
        super(LIF_Motorneuron, self).__init__(
            neuron_type='lif_motoneuron')
        #: Neuron constants
        #: Parameters of INaP
        self.g_nap = 10.0
        self.e_nap = 55.0
        self.am1_nap = 0.32
        self.am2_nap = -51.90
        self.am3_nap = 0.25
        self.bm1_nap = -0.280
        self.bm2_nap = -24.90
        self.bm3_nap = -0.2
        self.ah1_nap = 0.1280
        self.ah2_nap = -48.0
        self.ah3_nap = 0.0556
        self.bh1_nap = 4.0
        self.bh2_nap = -25.0
        self.bh3_nap = 0.20

        #: Parameters of IK
        self.g_k = 2.0
        self.e_k = -80.0
        self.am1_k = 0.0160
        self.am2_k = -29.90
        self.am3_k = 0.20
        self.bm1_k = 0.250
        self.bm2_k = -45.0
        self.bm3_k = 0.025

        #: Parameters of Iq
        self.g_q = 12.0
        self.e_q = -80.0
        self.vhm_q = -30.0
        self.gamma_q = -0.6
        self.r_q = 0.0005

        #: Parameters of Ileak
        self.g_leak = 0.8
        self.e_leak = -70.0

        #: Parameters of Isyn
        self.g_syn = 0.1
        self.e_syn = 0.0
        self.v_hs = -43.0
        self.gamma_s = -0.42

        #: Parameters of Iapp
        self.g_app = 0.19
        self.e_app = 0.0

        #: Other constants
        self.c_m = 1.0

    def ode(self, time, state, v_syn):
        """
        Parameters
        ----------
        time : <float>
            Current simulation time step
        states : <np.array>
            Neuron states
                * V : Membrane potential
                * h : Inactivation variable
        v_syn : <np.array>
            External synaptic membrane potentials
        Returns
        -------
        out : <np.array>
            Rate of change of neuron membrane potential

        """
        _V = state[0]  #: Membrane potential
        _m_nap = state[1]  #: Inactivation variable
        _h_nap = state[2]  #: Inactivation variable
        _m_k = state[3]  #: Inactivation variable
        _m_q = state[4]  #: Inactivation variable

        #: Helper functions
        def ix_func(g_x, e_x, m_x, h_x=1.):
            """ Compute the current."""
            p = 1
            return g_x*(m_x**p)*h_x*(_V - e_x)

        def inact_func(y, alpha_y, beta_y):
            """
            Compute the differential of inactivating functions
            """
            return alpha_y*(1-y) - beta_y*y

        def v_func(gam, v_h, v_ext=None):
            if v_ext is None:
                return 1. / (1 + np.exp(gam*(_V - v_h)))
            else:
                return 1. / (1 + np.exp(gam*(v_ext - v_h)))

        #: Inap
        alpha_m_nap = (self.am1_nap*(self.am2_nap - _V))/(
            np.exp(self.am3_nap*(self.am2_nap - _V)) - 1)

        beta_m_nap = (self.bm1_nap*(self.bm2_nap - _V))/(
            np.exp(self.bm3_nap*(self.bm2_nap - _V)) - 1)

        alpha_h_nap = self.ah1_nap*np.exp(self.ah3_nap*(self.ah2_nap - _V))

        beta_h_nap = (self.bh1_nap)/(
            np.exp(self.bh3_nap*(self.bh2_nap - _V)) + 1)

        I_nap = ix_func(self.g_nap, self.e_nap, _m_nap, _h_nap)

        #: Ik
        alpha_m_k = (self.am1_k*(self.am2_k - _V))/(
            np.exp(self.am3_k*(self.am2_k - _V)) - 1)

        beta_m_k = self.bm1_k*np.exp(
            self.bm3_k*(self.bm2_k - _V))

        I_k = ix_func(self.g_k, self.e_k, _m_k)

        #: Iq
        # TODO: Check the equation : Sq
        m_q_inf = 1./(1 + np.exp(self.gamma_q*(_V - self.vhm_q)))

        alpha_m_q = m_q_inf*self.r_q

        beta_m_q = (1 - m_q_inf)*self.r_q

        I_q = ix_func(self.g_q, self.e_q, _m_q)

        #: Ileak
        I_leak = self.g_leak*(_V - self.e_leak)

        #: Iapp
        I_app = self.g_app*(_V - self.e_app)

        #: Isyn
        I_syn = np.sum(self.g_syn*v_func(
            self.gamma_s, self.v_hs, v_syn)*(_V - self.e_syn))

        #: dV
        _dV = -(I_nap + I_k + I_q + I_leak + I_syn + I_app)/self.c_m

        _dm_nap = inact_func(_m_nap, alpha_m_nap, beta_m_nap)

        _dh_nap = inact_func(_h_nap, alpha_h_nap, beta_h_nap)

        _dm_k = inact_func(_m_k, alpha_m_k, beta_m_k)

        _dm_q = inact_func(_m_q, alpha_m_q, beta_m_q)

        return np.array([_dV, _dm_nap, _dh_nap, _dm_k, _dm_q])


def main():
    neuron1 = LIF_Interneuron()
    biolog.debug('Neuron type : {}'.format(neuron1.neuron_type))
    biolog.debug('Neuron ode out : {}'.format(
        neuron1.ode(0.0, np.array([-56., 0.0]), np.arange(0, -5, 1))))

    neuron2 = LIF_Motorneuron()
    biolog.debug('Neuron type : {}'.format(neuron2.neuron_type))
    biolog.debug('Neuron ode out : {}'.format(
        neuron2.ode(0.0, np.array([-56., 0.0, 0.0, 0.0, 0.0]), np.arange(
            0, -5, 1))))


if __name__ == '__main__':
    main()
