""" Options to configure the neural and network models """


from typing import List

from farms_core.options import Options


class NeuronOptions(Options):
    """ Base class for defining neuron options """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__()
        self.name: str = kwargs.pop("name")

        self.parameters: NeuronParameterOptions = kwargs.pop("parameters")
        self.visual: NeuronVisualOptions = kwargs.pop("visual")
        self.state: NeuronStateOptions = kwargs.pop("state", None)

        self.nstates: int = 0
        self.nparameters: int = 0
        self.ninputs: int = 0


class NetworkOptions(Options):
    """ Base class for neural network options """

    def __init__(self, **kwargs):
        super().__init__()
        self.directed: bool = kwargs.pop("directed")
        self.multigraph: bool = kwargs.pop("multigraph")
        self.name: str = kwargs.pop("name")


class NeuronParameterOptions(Options):
    """ Base class for neuron specific parameters """

    def __init__(self):
        super().__init__()


class NeuronStateOptions(Options):
    """ Base class for neuron specific state options """
    def __init__(self, **kwargs):
        super().__init__()
        self.initial: List[float] = kwargs.pop("initial")


class NeuronVisualOptions(Options):
    """ Base class for neuron visualization parameters """

    def __init__(self, **kwargs):
        super().__init__()
        self.position: List[float] = kwargs.pop("position")
        self.color: List[float] = kwargs.pop("color")
        self.layer: str = kwargs.pop("layer")


class LIDannerParameterOptions(NeuronParameterOptions):
    """ Class to define the parameters of Leaky integrator danner neuron model """

    def __init__(self, **kwargs):
        super().__init__()
        self.c_m = kwargs.pop("c_m")                        # pF
        self.g_leak = kwargs.pop("g_leak")                  # nS
        self.e_leak = kwargs.pop("e_leak")                  # mV
        self.v_max = kwargs.pop("v_max")                    # mV
        self.v_thr = kwargs.pop("v_thr")                    # mV
        self.g_syn_e = kwargs.pop("g_syn_e")                # nS
        self.g_syn_i = kwargs.pop("g_syn_i")                # nS
        self.e_syn_e = kwargs.pop("e_syn_e")                # mV
        self.e_syn_i = kwargs.pop("e_syn_i")                # mV
        self.m_e = kwargs.pop("m_e")                        # -
        self.m_i = kwargs.pop("m_i")                        # -
        self.b_e = kwargs.pop("b_e")                        # -
        self.b_i = kwargs.pop("b_i")                        # -

    @classmethod
    def defaults(cls):
        """ Get the default parameters for LI Danner Neuron model """

        options = {}

        options["c_m"] = 10.0
        options["g_leak"] = 2.8
        options["e_leak"] = -60.0
        options["v_max"] = 0.0
        options["v_thr"] = -50.0
        options["g_syn_e"] = 10.0
        options["g_syn_i"] = 10.0
        options["e_syn_e"] = -10.0
        options["e_syn_i"] = -75.0
        options["m_e"] = 0.0
        options["m_i"] = 0.0
        options["b_e"] = 0.0
        options["b_i"] = 0.0

        return cls(**options)


class LIDannerNeuronOptions(NeuronOptions):
    """ Class to define the properties of Leaky integrator danner neuron model """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__(
            name=kwargs.pop("name"),
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
        )
        self.nstates = 2
        self.nparameters = 13

        self.ninputs = kwargs.pop("ninputs")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')
