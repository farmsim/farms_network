""" Options to configure the neural and network models """


import time
from typing import Any, Dict, Iterable, List, Self, Type

from farms_core import pylog
from farms_core.options import Options
from farms_network.models import EdgeTypes, Models


###########################
# Node Base Class Options #
###########################
class NodeOptions(Options):
    """ Base class for defining node options """
    MODEL = Models.BASE

    def __init__(self, **kwargs: Any):
        """ Initialize """
        super().__init__()
        self.name: str = kwargs.pop("name")
        model = kwargs.pop("model", NodeOptions.MODEL)
        if isinstance(model, Models):
            model = Models.to_str(model)
        elif not isinstance(model, str):
            raise TypeError(
                f"{model} is of {type(model)}. Needs to {type(Models)} or {type(str)}"
            )
        self.model: str = model
        self.parameters: NodeParameterOptions = kwargs.pop("parameters", None)
        self.visual: NodeVisualOptions = NodeVisualOptions.from_options(
            kwargs.pop("visual")
        )
        self.state: NodeStateOptions = kwargs.pop("state", None)
        self.noise: NoiseOptions = kwargs.pop("noise", None)

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = kwargs.pop("parameters")
        options["visual"] = kwargs.pop("visual")
        options["state"] = kwargs.pop("state")
        options["noise"] = kwargs.pop("noise")
        return cls(**options)

    def __eq__(self, other):
        if isinstance(other, NodeOptions):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self):
        return hash(self.name)  # Hash based on the node name (or any unique property)

    def __str__(self) -> str:
        attrs_str = ',\n'.join(f'{attr}={getattr(self, attr)}' for attr in self.keys())
        return f"{self.__class__.__name__}(\n{attrs_str}\n)"

    def __repr__(self) -> str:
        return self.__str__()


class NodeParameterOptions(Options):
    """ Base class for node specific parameters """

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        return cls(**kwargs)


class NodeStateOptions(Options):
    """ Base class for node specific state options """

    STATE_NAMES: List[str] = []  # Override in subclasses

    def __init__(self, **kwargs):
        super().__init__()
        self.initial: List[float] = kwargs.pop("initial")
        self.names: List[str] = kwargs.pop("names")

        if kwargs:
            raise ValueError(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_kwargs(cls, **kwargs):
        """ From node specific name-value kwargs """
        initial = [
            kwargs.pop(name)
            for name in cls.STATE_NAMES
        ]
        if kwargs:
            raise ValueError(f'Unknown kwargs: {kwargs}')
        return cls(initial=initial)

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        return cls(**kwargs)


class NodeLogOptions(Options):
    """ Log options for the node level """

    def __init__(self, buffer_size: int, enable: bool, **kwargs):
        super().__init__(**kwargs)

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        return cls(**kwargs)


class NodeVisualOptions(Options):
    """ Base class for node visualization parameters """

    def __init__(self, **kwargs):
        super().__init__()
        self.position: List[float] = kwargs.pop("position", [0.0, 0.0, 0.0])
        self.radius: float = kwargs.pop("radius", 1.0)
        self.color: List[float] = kwargs.pop("color", [1.0, 0.0, 0.0])
        self.label: str = kwargs.pop("label", "n")
        self.layer: str = kwargs.pop("layer", "background")
        self.latex: dict = kwargs.pop("latex", "{}")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        return cls(**kwargs)


################
# Edge Options #
################
class EdgeOptions(Options):
    """ Base class for defining edge options between nodes """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__()
        self.source: str = kwargs.pop("source")
        self.target: str = kwargs.pop("target")
        self.weight: float = kwargs.pop("weight")
        self.type = EdgeTypes.to_str(kwargs.pop("type"))
        self.parameters: EdgeParameterOptions = kwargs.pop(
            "parameters", EdgeParameterOptions()
        )

        self.visual: EdgeVisualOptions = kwargs.pop("visual")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    def __eq__(self, other):
        if isinstance(other, EdgeOptions):
            return (
                (self.source == other.source) and
                (self.target == other.target)
            )
        return False

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """

        options = {}
        options["source"] = kwargs["source"]
        options["target"] = kwargs["target"]
        options["weight"] = kwargs["weight"]
        options["type"] = kwargs["type"]
        options["parameters"] = EdgeParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = EdgeVisualOptions.from_options(
            kwargs["visual"]
        )
        return cls(**options)

    def __str__(self) -> str:
        attrs_str = ',\n'.join(f'{attr}={getattr(self, attr)}' for attr in self.keys())
        return f"{self.__class__.__name__}(\n{attrs_str}\n)"

    def __repr__(self) -> str:
        return self.__str__()


class EdgeParameterOptions(Options):
    """ Base class for edge specific parameters """

    def __init__(self, **kwargs):
        super().__init__()
        self.model: str = kwargs.pop("model", None)

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        return cls(**kwargs)


class EdgeVisualOptions(Options):
    """ Base class for edge visualization parameters """

    def __init__(self, **kwargs):
        super().__init__()
        self.color: List[float] = kwargs.pop("color", [1.0, 0.0, 0.0])
        self.alpha: float = kwargs.pop("alpha", 1.0)
        self.label: str = kwargs.pop("label", "")
        self.layer: str = kwargs.pop("layer", "background")
        self.latex: dict = kwargs.pop("latex", "{}")

        # New options for FancyArrowPatch compatibility
        self.arrowstyle: str = kwargs.pop("arrowstyle", "->")
        self.connectionstyle: str = kwargs.pop("connectionstyle", "arc3,rad=0.1")
        self.linewidth: float = kwargs.pop("linewidth", 1.5)
        self.edgecolor: List[float] = kwargs.pop("edgecolor", [0.0, 0.0, 0.0])

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        return cls(**kwargs)


#################
# Noise Options #
#################
class NoiseOptions(Options):
    """ Base class for node noise options """

    NOISE_TYPES = ("additive",)
    NOISE_MODELS = ("white", "ornstein_uhlenbeck")

    def __init__(self, **kwargs):
        super().__init__()
        self.type = kwargs.pop("type", NoiseOptions.NOISE_TYPES[0])
        assert self.type.lower() in NoiseOptions.NOISE_TYPES
        self.model = kwargs.pop("model", None)
        assert self.model.lower() in NoiseOptions.NOISE_MODELS
        self.is_stochastic = kwargs.pop("is_stochastic")

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        noise_models = {
            cls.NOISE_MODELS[0]: NoiseOptions,
            cls.NOISE_MODELS[1]: OrnsteinUhlenbeckOptions
        }
        noise_model = kwargs.pop("model")
        return noise_models[noise_model].from_options(kwargs)


class OrnsteinUhlenbeckOptions(NoiseOptions):
    """ Options to  OrnsteinUhlenbeckOptions """

    def __init__(self, **kwargs):
        """ Initialize """
        model = NoiseOptions.NOISE_MODELS[1]
        is_stochastic = True
        super().__init__(model=model, is_stochastic=is_stochastic)
        self.mu: float = kwargs.pop("mu")
        self.sigma: float = kwargs.pop("sigma")
        self.tau: float = kwargs.pop("tau")

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        options = {}
        options["mu"] = kwargs.pop("mu")
        options["sigma"] = kwargs.pop("sigma")
        options["tau"] = kwargs.pop("tau")
        return cls(**options)

    @classmethod
    def defaults(cls, **kwargs: Dict):
        """ From options """
        options = {}
        options["mu"] = kwargs.pop("mu", 0.0)
        options["sigma"] = kwargs.pop("sigma", 0.005)
        options["tau"] = kwargs.pop("tau", 10.0)
        return cls(**options)


#######################
# Relay Model Options #
#######################
class RelayNodeOptions(NodeOptions):
    """ Class to define the properties of Relay node model """
    MODEL = Models.RELAY

    def __init__(self, **kwargs):
        """ Initialize """
        state = kwargs.pop("state", None)
        parameters = kwargs.pop("parameters", None)

        assert state is None
        assert parameters is None
        super().__init__(
            name=kwargs.pop("name"),
            model=RelayNodeOptions.MODEL,
            parameters=parameters,
            visual=kwargs.pop("visual"),
            state=state,
            noise=kwargs.pop("noise"),
        )
        self._nstates = 0
        self._nparameters = 0

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = None
        options["visual"] = NodeVisualOptions.from_options(kwargs["visual"])
        options["noise"] = kwargs.pop("noise", None)
        return cls(**options)


class RelayEdgeOptions(EdgeOptions):
    """ Relay edge options """
    MODEL = Models.RELAY

    def __init__(self, **kwargs):
        """ Initialize """

        super().__init__(
            model=RelayEdgeOptions.MODEL,
            source=kwargs.pop("source"),
            target=kwargs.pop("target"),
            weight=kwargs.pop("weight"),
            type=kwargs.pop("type"),
            parameters=None,
            visual=kwargs.pop("visual"),
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """

        options = {}
        options["source"] = kwargs["source"]
        options["target"] = kwargs["target"]
        options["weight"] = kwargs["weight"]
        options["type"] = kwargs["type"]
        options["visual"] = EdgeVisualOptions.from_options(kwargs["visual"])
        return cls(**options)


########################
# Linear Model Options #
########################
class LinearNodeOptions(NodeOptions):
    """ Class to define the properties of Linear node model """
    MODEL = Models.LINEAR

    def __init__(self, **kwargs):
        """ Initialize """

        state = kwargs.pop("state", None)
        assert state is None
        super().__init__(
            name=kwargs.pop("name"),
            model=LinearNodeOptions.MODEL,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=state,
            noise=kwargs.pop("noise"),
        )
        self._nstates = 0
        self._nparameters = 2

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = LinearParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = NodeVisualOptions.from_options(
            kwargs["visual"]
        )
        options["noise"] = kwargs.pop("noise", None)
        return cls(**options)


class LinearParameterOptions(NodeParameterOptions):

    def __init__(self, **kwargs):
        super().__init__()
        self.slope = kwargs.pop("slope")
        self.bias = kwargs.pop("bias")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for Linear Node model """

        options = {}
        options["slope"] = kwargs.pop("slope", 1.0)
        options["bias"] = kwargs.pop("bias", 0.0)
        return cls(**options)


class LinearEdgeOptions(EdgeOptions):
    """ Linear edge options """

    def __init__(self, **kwargs):
        """ Initialize """

        super().__init__(
            model=Models.LINEAR,
            source=kwargs.pop("source"),
            target=kwargs.pop("target"),
            weight=kwargs.pop("weight"),
            type=kwargs.pop("type"),
            visual=kwargs.pop("visual"),
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """

        options = {}
        options["source"] = kwargs["source"]
        options["target"] = kwargs["target"]
        options["weight"] = kwargs["weight"]
        options["type"] = kwargs["type"]
        options["parameters"] = None
        options["visual"] = EdgeVisualOptions.from_options(kwargs["visual"])
        return cls(**options)


######################
# ReLU Model Options #
######################
class ReLUNodeOptions(NodeOptions):
    """ Class to define the properties of ReLU node model """

    MODEL = Models.RELU

    def __init__(self, **kwargs):
        """ Initialize """

        state = kwargs.pop("state", None)
        assert state is None
        super().__init__(
            name=kwargs.pop("name"),
            model=ReLUNodeOptions.MODEL,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=state,
            noise=kwargs.pop("noise"),
        )
        self._nstates = 0
        self._nparameters = 3

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = ReLUParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = NodeVisualOptions.from_options(
            kwargs["visual"]
        )
        options["state"] = None
        options["noise"] = kwargs.pop("noise", None)
        return cls(**options)


class ReLUParameterOptions(NodeParameterOptions):

    def __init__(self, **kwargs):
        super().__init__()
        self.gain = kwargs.pop("gain")
        self.sign = kwargs.pop("sign")
        self.offset = kwargs.pop("offset")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for ReLU Node model """
        options = {}
        options["gain"] = kwargs.pop("gain", 1.0)
        options["sign"] = kwargs.pop("sign", 1)
        options["offset"] = kwargs.pop("offset", 0.0)

        return cls(**options)


class ReLUEdgeOptions(EdgeOptions):
    """ ReLU edge options """

    def __init__(self, **kwargs):
        """ Initialize """

        super().__init__(
            model=Models.RELU,
            source=kwargs.pop("source"),
            target=kwargs.pop("target"),
            weight=kwargs.pop("weight"),
            type=kwargs.pop("type"),
            visual=kwargs.pop("visual"),
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """

        options = {}
        options["source"] = kwargs["source"]
        options["target"] = kwargs["target"]
        options["weight"] = kwargs["weight"]
        options["type"] = kwargs["type"]
        options["parameters"] = None
        options["visual"] = EdgeVisualOptions.from_options(kwargs["visual"])
        return cls(**options)


############################################
# Phase-Amplitude Oscillator Model Options #
############################################
class OscillatorNodeOptions(NodeOptions):
    """ Class to define the properties of Oscillator node model """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "oscillator"
        super().__init__(
            name=kwargs.pop("name"),
            model=model,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
            noise=kwargs.pop("noise"),
        )
        self._nstates = 3
        self._nparameters = 3

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = OscillatorNodeParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = NodeVisualOptions.from_options(
            kwargs["visual"]
        )
        options["state"] = OscillatorStateOptions.from_options(
            kwargs["state"]
        )
        options["noise"] = None
        if kwargs["noise"] is not None:
            options["noise"] = NoiseOptions.from_options(
                kwargs["noise"]
            )
        return cls(**options)


class OscillatorNodeParameterOptions(NodeParameterOptions):

    def __init__(self, **kwargs):
        super().__init__()
        self.intrinsic_frequency = kwargs.pop("intrinsic_frequency")  # Hz
        self.nominal_amplitude = kwargs.pop("nominal_amplitude")      #
        self.amplitude_rate = kwargs.pop("amplitude_rate")            #

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for Oscillator Node model """

        options = {}

        options["intrinsic_frequency"] = kwargs.pop("intrinsic_frequency", 1.0)
        options["nominal_amplitude"] = kwargs.pop("nominal_amplitude", 1.0)
        options["amplitude_rate"] = kwargs.pop("amplitude_rate", 1.0)

        return cls(**options)


class OscillatorStateOptions(NodeStateOptions):
    """ Oscillator node state options """

    STATE_NAMES = ["phase", "amplitude_0", "amplitude"]

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial"),
            names=OscillatorStateOptions.STATE_NAMES
        )
        assert len(self.initial) == 3, f"Number of initial states {len(self.initial)} should be 3"


class OscillatorEdgeOptions(EdgeOptions):
    """ Oscillator edge options """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "oscillator"
        super().__init__(
            model=model,
            source=kwargs.pop("source"),
            target=kwargs.pop("target"),
            weight=kwargs.pop("weight"),
            type=kwargs.pop("type"),
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    def __eq__(self, other):
        if isinstance(other, EdgeOptions):
            return (
                (self.source == other.source) and
                (self.target == other.target)
            )
        return False

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """

        options = {}
        options["source"] = kwargs["source"]
        options["target"] = kwargs["target"]
        options["weight"] = kwargs["weight"]
        options["type"] = kwargs["type"]
        options["parameters"] = OscillatorEdgeParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = EdgeVisualOptions.from_options(kwargs["visual"])
        return cls(**options)


class OscillatorEdgeParameterOptions(EdgeParameterOptions):
    """ Oscillator edge parameter options """

    def __init__(self, **kwargs):
        super().__init__()
        self.phase_difference = kwargs.pop("phase_difference")   # radians

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for Oscillator Node model """

        options = {}
        options["phase_difference"] = kwargs.pop("phase_difference", 0.0)
        return cls(**options)


#################################
# Hopf-Oscillator Model Options #
#################################
class HopfOscillatorNodeOptions(NodeOptions):
    """ Class to define the properties of HopfOscillator node model """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "hopf_oscillator"
        super().__init__(
            name=kwargs.pop("name"),
            model=model,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
            noise=kwargs.pop("noise"),
        )
        self._nstates = 2
        self._nparameters = 4

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = HopfOscillatorNodeParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = NodeVisualOptions.from_options(
            kwargs["visual"]
        )
        options["state"] = HopfOscillatorStateOptions.from_options(
            kwargs["state"]
        )
        options["noise"] = None
        if kwargs["noise"] is not None:
            options["noise"] = NoiseOptions.from_options(
                kwargs["noise"]
            )
        return cls(**options)


class HopfOscillatorNodeParameterOptions(NodeParameterOptions):

    def __init__(self, **kwargs):
        super().__init__()
        self.mu = kwargs.pop("mu")
        self.omega = kwargs.pop("omega")
        self.alpha = kwargs.pop("alpha")
        self.beta = kwargs.pop("beta")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for HopfOscillator Node model """

        options = {}

        options["mu"] = kwargs.pop("mu", 0.1)
        options["omega"] = kwargs.pop("omega", 0.1)
        options["alpha"] = kwargs.pop("alpha", 1.0)
        options["beta"] = kwargs.pop("beta", 1.0)

        return cls(**options)


class HopfOscillatorStateOptions(NodeStateOptions):
    """ HopfOscillator node state options """

    STATE_NAMES = ["x", "y"]

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial"),
            names=HopfOscillatorStateOptions.STATE_NAMES
        )
        assert len(self.initial) == 2, f"Number of initial states {len(self.initial)} should be 2"


##################################
# Leaky Integrator Model Options #
##################################
class LeakyIntegratorNodeOptions(NodeOptions):
    """ Class to define the properties for standard leaky integrator model """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "leaky_integrator"
        super().__init__(
            name=kwargs.pop("name"),
            model=model,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
            noise=kwargs.pop("noise"),
        )
        self._nstates = 1
        self._nparameters = 3

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = LeakyIntegratorParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = NodeVisualOptions.from_options(
            kwargs["visual"]
        )
        options["state"] = LeakyIntegratorStateOptions.from_options(
            kwargs["state"]
        )
        options["noise"] = None
        if kwargs["noise"] is not None:
            options["noise"] = NoiseOptions.from_options(
                kwargs["noise"]
            )
        return cls(**options)


class LeakyIntegratorParameterOptions(NodeParameterOptions):
    """
    Class to define the parameters of Leaky Integrator model.

    Attributes:
        tau (float): Time constant.
        bias (float)
        D (float)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.tau = kwargs.pop("tau")
        self.bias = kwargs.pop("bias")
        self.D = kwargs.pop("D")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for LI Danner Node model """

        options = {}

        options["tau"] = kwargs.pop("tau", 0.1)
        options["bias"] = kwargs.pop("bias", -2.75)
        options["D"] = kwargs.pop("D", 1.0)

        return cls(**options)


class LeakyIntegratorStateOptions(NodeStateOptions):
    """ LeakyIntegrator node state options """

    STATE_NAMES = ["m",]

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial"),
            names=LeakyIntegratorStateOptions.STATE_NAMES
        )
        assert len(self.initial) == 1, f"Number of initial states {len(self.initial)} should be 1"


#########################################
# Leaky Integrator Danner Model Options #
#########################################
class LIDannerNodeOptions(NodeOptions):
    """ Class to define the properties of Leaky integrator danner node model """

    MODEL = Models.LI_DANNER

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__(
            name=kwargs.pop("name"),
            model=LIDannerNodeOptions.MODEL,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
            noise=kwargs.pop("noise"),
        )
        self._nstates = 2
        self._nparameters = 10

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = LIDannerNodeParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = NodeVisualOptions.from_options(
            kwargs["visual"]
        )
        options["state"] = LIDannerStateOptions.from_options(
            kwargs["state"]
        )
        options["noise"] = None
        if kwargs["noise"] is not None:
            options["noise"] = NoiseOptions.from_options(
                kwargs["noise"]
            )
        return cls(**options)


class LIDannerNodeParameterOptions(NodeParameterOptions):
    """
    Class to define the parameters of Leaky Integrator Danner node model.

    Attributes:
        c_m (float): Membrane capacitance (in pF).
        g_leak (float): Leak conductance (in nS).
        e_leak (float): Leak reversal potential (in mV).
        v_max (float): Maximum voltage (in mV).
        v_thr (float): Threshold voltage (in mV).
        g_syn_e (float): Excitatory synaptic conductance (in nS).
        g_syn_i (float): Inhibitory synaptic conductance (in nS).
        e_syn_e (float): Excitatory synaptic reversal potential (in mV).
        e_syn_i (float): Inhibitory synaptic reversal potential (in mV).
        tau_ch (float): Cholinergic time constant (in mS)
    """

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
        self.tau_ch = kwargs.pop("tau_ch", 5.0)                  # tau-cholinergic
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for LI Danner Node model """

        options = {}
        options["c_m"] = kwargs.pop("c_m",  10.0)
        options["g_leak"] = kwargs.pop("g_leak",  2.8)
        options["e_leak"] = kwargs.pop("e_leak",  -60.0)
        options["v_max"] = kwargs.pop("v_max",  0.0)
        options["v_thr"] = kwargs.pop("v_thr",  -50.0)
        options["g_syn_e"] = kwargs.pop("g_syn_e",  10.0)
        options["g_syn_i"] = kwargs.pop("g_syn_i",  10.0)
        options["e_syn_e"] = kwargs.pop("e_syn_e",  -10.0)
        options["e_syn_i"] = kwargs.pop("e_syn_i",  -75.0)
        options["tau_ch"] = kwargs.pop("tau_ch",  5.0)
        return cls(**options)


class LIDannerStateOptions(NodeStateOptions):
    """ LI Danner node state options """

    STATE_NAMES = ["v", "a"]

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial", [-60.0, 0.0]),
            names=LIDannerStateOptions.STATE_NAMES
        )
        # assert len(self.initial) == 2, f"Number of initial states {len(self.initial)} should be 2"


class LIDannerEdgeOptions(EdgeOptions):
    """ LIDanner edge options """

    MODEL = Models.LI_DANNER

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__(
            model=LIDannerEdgeOptions.MODEL,
            source=kwargs.pop("source"),
            target=kwargs.pop("target"),
            weight=kwargs.pop("weight"),
            type=kwargs.pop("type"),
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """

        options = {}
        options["source"] = kwargs["source"]
        options["target"] = kwargs["target"]
        options["weight"] = kwargs["weight"]
        options["type"] = kwargs["type"]
        options["parameters"] = None
        options["visual"] = EdgeVisualOptions.from_options(kwargs["visual"])
        return cls(**options)


##################################################
# Leaky Integrator With NaP Danner Model Options #
##################################################
class LINaPDannerNodeOptions(NodeOptions):
    """ Class to define the properties of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "li_nap_danner"
        super().__init__(
            name=kwargs.pop("name"),
            model=model,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
            noise=kwargs.pop("noise"),
        )
        self._nstates = 2
        self._nparameters = 19

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ Load from options """
        options = {}
        options["name"] = kwargs.pop("name")
        options["parameters"] = LINaPDannerNodeParameterOptions.from_options(
            kwargs["parameters"]
        )
        options["visual"] = NodeVisualOptions.from_options(
            kwargs["visual"]
        )
        options["state"] = LINaPDannerStateOptions.from_options(
            kwargs["state"]
        )
        options["noise"] = None
        if kwargs["noise"] is not None:
            options["noise"] = NoiseOptions.from_options(
                kwargs["noise"]
            )
        return cls(**options)


class LINaPDannerNodeParameterOptions(NodeParameterOptions):
    """ Class to define the parameters of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        super().__init__()

        self.c_m = kwargs.pop("c_m")                        # pF
        self.g_nap = kwargs.pop("g_nap")                    # nS
        self.e_na = kwargs.pop("e_na")                      # mV
        self.v1_2_m = kwargs.pop("v1_2_m")                  # mV
        self.k_m = kwargs.pop("k_m")                        #
        self.v1_2_h = kwargs.pop("v1_2_h")                  # mV
        self.k_h = kwargs.pop("k_h")                        #
        self.v1_2_t = kwargs.pop("v1_2_t")                  # mV
        self.k_t = kwargs.pop("k_t")                        #
        self.g_leak = kwargs.pop("g_leak")                  # nS
        self.e_leak = kwargs.pop("e_leak")                  # mV
        self.tau_0 = kwargs.pop("tau_0")                    # mS
        self.tau_max = kwargs.pop("tau_max")                # mS
        self.v_max = kwargs.pop("v_max")                    # mV
        self.v_thr = kwargs.pop("v_thr")                    # mV
        self.g_syn_e = kwargs.pop("g_syn_e")                # nS
        self.g_syn_i = kwargs.pop("g_syn_i")                # nS
        self.e_syn_e = kwargs.pop("e_syn_e")                # mV
        self.e_syn_i = kwargs.pop("e_syn_i")                # mV

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for LI NaP Danner Node model """

        options = {}

        options["c_m"] = kwargs.pop("c_m", 10.0)                  # pF
        options["g_nap"] = kwargs.pop("g_nap", 4.5)               # nS
        options["e_na"] = kwargs.pop("e_na", 50.0)                # mV
        options["v1_2_m"] = kwargs.pop("v1_2_m", -40.0)           # mV
        options["k_m"] = kwargs.pop("k_m", -6.0)                  #
        options["v1_2_h"] = kwargs.pop("v1_2_h", -45.0)           # mV
        options["k_h"] = kwargs.pop("k_h", 4.0)                   #
        options["v1_2_t"] = kwargs.pop("v1_2_t", -35.0)           # mV
        options["k_t"] = kwargs.pop("k_t", 15.0)                  #
        options["g_leak"] = kwargs.pop("g_leak", 4.5)             # nS
        options["e_leak"] = kwargs.pop("e_leak", -62.5)           # mV
        options["tau_0"] = kwargs.pop("tau_0", 80.0)              # mS
        options["tau_max"] = kwargs.pop("tau_max", 160.0)         # mS
        options["v_max"] = kwargs.pop("v_max", 0.0)               # mV
        options["v_thr"] = kwargs.pop("v_thr", -50.0)             # mV
        options["g_syn_e"] = kwargs.pop("g_syn_e",  10.0)
        options["g_syn_i"] = kwargs.pop("g_syn_i",  10.0)
        options["e_syn_e"] = kwargs.pop("e_syn_e",  -10.0)
        options["e_syn_i"] = kwargs.pop("e_syn_i",  -75.0)

        return cls(**options)


class LINaPDannerStateOptions(NodeStateOptions):
    """ LI Danner node state options """

    STATE_NAMES = ["v", "h"]

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial"),
            names=LINaPDannerStateOptions.STATE_NAMES
        )
        assert len(self.initial) == 2, f"Number of initial states {len(self.initial)} should be 2"


class LINaPDannerEdgeOptions(EdgeOptions):
    """ LINaPDanner edge options """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "li_danner"
        super().__init__(
            model=model,
            source=kwargs.pop("source"),
            target=kwargs.pop("target"),
            weight=kwargs.pop("weight"),
            type=kwargs.pop("type"),
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """

        options = {}
        options["source"] = kwargs["source"]
        options["target"] = kwargs["target"]
        options["weight"] = kwargs["weight"]
        options["type"] = kwargs["type"]
        options["parameters"] = None
        options["visual"] = EdgeVisualOptions.from_options(kwargs["visual"])
        return cls(**options)


####################
# Izhikevich Model #
####################
class IzhikevichNodeOptions(NodeOptions):
    """ Class to define the properties of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "izhikevich"
        super().__init__(
            name=kwargs.pop("name"),
            model=model,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
            noise=kwargs.pop("noise"),
        )
        self._nstates = 2
        self._nparameters = 5

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class IzhikevichParameterOptions(NodeParameterOptions):
    """ Class to define the parameters of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        super().__init__()

        self.recovery_time = kwargs.pop("recovery_time")    # pF
        self.recovery_sensitivity = kwargs.pop("recovery_sensitivity")  # nS
        self.membrane_reset = kwargs.pop("membrane_reset")              # mV
        self.recovery_reset = kwargs.pop("recovery_reset")              # mV
        self.membrane_threshold = kwargs.pop("membrane_threshold")      # mV

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for LI NaP Danner Node model """

        options = {}

        options["recovery_time"] = kwargs.pop("recovery_time", 0.02)               # pF
        options["recovery_sensitivity"] = kwargs.pop("recovery_sensitivity", 0.2)  # nS
        options["membrane_reset"] = kwargs.pop("membrane_reset", -65.0)            # mV
        options["recovery_reset"] = kwargs.pop("recovery_reset", 2)                # mV
        options["membrane_threshold"] = kwargs.pop("membrane_threshold", 30.0)     # mV

        return cls(**options)


class IzhikevichStateOptions(NodeStateOptions):
    """ LI Danner node state options """

    STATE_NAMES = ["v", "u"]

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial"),
            names=IzhikevichStateOptions.STATE_NAMES
        )
        assert len(self.initial) == 2, f"Number of initial states {len(self.initial)} should be 2"


##############################
# Network Base Class Options #
##############################
class NetworkOptions(Options):
    """ Base class for neural network options """

    NODE_TYPES: Dict[Models, Type] = {
        Models.RELAY: RelayNodeOptions,
        Models.LINEAR: LinearNodeOptions,
        Models.RELU: ReLUNodeOptions,
        Models.OSCILLATOR: OscillatorNodeOptions,
        # Models.HOPF_OSCILLATOR: HopfOscillatorNodeOptions,
        # Models.MORPHED_OSCILLATOR: MorphedOscillatorNodeOptions,
        # Models.MATSUOKA: MatsuokaNodeOptions,
        # Models.FITZHUGH_NAGUMO: FitzhughNagumoNodeOptions,
        # Models.MORRIS_LECAR: MorrisLecarNodeOptions,
        # Models.LEAKY_INTEGRATOR: LeakyIntegratorNodeOptions,
        Models.LI_DANNER: LIDannerNodeOptions,
        Models.LI_NAP_DANNER: LINaPDannerNodeOptions,
        # Models.LI_DAUN: LIDaunNodeOptions,
        # Models.HH_DAUN: HHDaunNodeOptions,
    }

    EDGE_TYPES: Dict[Models, Type] = {
        Models.RELAY: RelayEdgeOptions,
        Models.LINEAR: LinearEdgeOptions,
        Models.RELU: ReLUEdgeOptions,
        Models.OSCILLATOR: OscillatorEdgeOptions,
        # Models.HOPF_OSCILLATOR: HopfOscillatorEdgeOptions,
        # Models.MORPHED_OSCILLATOR: MorphedOscillatorEdgeOptions,
        # Models.MATSUOKA: MatsuokaEdgeOptions,
        # Models.FITZHUGH_NAGUMO: FitzhughNagumoEdgeOptions,
        # Models.MORRIS_LECAR: MorrisLecarEdgeOptions,
        # Models.LEAKY_INTEGRATOR: LeakyIntegratorEdgeOptions,
        Models.LI_DANNER: LIDannerEdgeOptions,
        Models.LI_NAP_DANNER: LINaPDannerEdgeOptions,
        # Models.LI_DAUN: LIDaunEdgeOptions,
        # Models.HH_DAUN: HHDaunEdgeOptions,
    }

    def __init__(self, **kwargs):
        super().__init__()

        # Default properties to make it compatible with networkx
        # seed
        self.directed: bool = kwargs.pop("directed", True)
        self.multigraph: bool = kwargs.pop("multigraph", False)
        self.graph: dict = kwargs.pop("graph", {"name": ""})
        self.units = kwargs.pop("units", None)
        self.logs: NetworkLogOptions = kwargs.pop("logs")
        self.random_seed: int = kwargs.pop("random_seed", time.time_ns())

        self.integration = kwargs.pop(
            "integration", IntegrationOptions.defaults()
        )

        self.nodes: List[NodeOptions] = kwargs.pop("nodes", [])
        self.edges: List[EdgeOptions] = kwargs.pop("edges", [])

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs):
        """ From options """
        options = {}
        options["directed"] = kwargs["directed"]
        options["multigraph"] = kwargs["multigraph"]
        options["graph"] = kwargs["graph"]
        options["units"] = kwargs["units"]
        # Log options
        options["logs"] = NetworkLogOptions.from_options(kwargs["logs"])
        # Integration options
        options["integration"] = IntegrationOptions.from_options(kwargs["integration"])
        # Nodes
        options["nodes"] = [
            cls.NODE_TYPES[node["model"]].from_options(node)
            for node in kwargs["nodes"]
        ]
        # Edges
        options["edges"] = [
            # cls.EDGE_TYPES[edge["model"]].from_options(edge)
            EdgeOptions.from_options(edge)
            for edge in kwargs["edges"]
        ]
        return cls(**options)

    def add_node(self, options: NodeOptions):
        """ Add a node if it does not already exist in the list """
        assert isinstance(options, NodeOptions), f"{type(options)} not an instance of NodeOptions"
        if options not in self.nodes:
            self.nodes.append(options)
        else:
            print(f"Node {options.name} already exists and will not be added again.")

    def add_nodes(self, options: Iterable[NodeOptions]):
        """ Add a collection of nodes """
        for node in options:
            self.add_node(node)

    def add_edge(self, options: EdgeOptions):
        """ Add a node if it does not already exist in the list """
        if (options.source in self.nodes) and (options.target in self.nodes):
            self.edges.append(options)
        else:
            missing_nodes = [
                "" if (options.source in self.nodes) else options.source,
                "" if (options.target in self.nodes) else options.target,
            ]
            pylog.debug(f"Missing node {*missing_nodes,} in Edge {options}")

    def add_edges(self, options: Iterable[EdgeOptions]):
        """ Add a collection of edges """
        for edge in options:
            self.add_edge(edge)

    def __add__(self, other: Self):
        """ Combine two network options """
        assert isinstance(other, NetworkOptions)
        for node in other.nodes:
            self.add_node(node)
        for edge in other.edges:
            self.add_edge(edge)
        return self


#################################
# Numerical Integration Options #
#################################
class IntegrationOptions(Options):
    """ Class to set the options for numerical integration """

    def __init__(self, **kwargs):
        super().__init__()

        self.timestep: float = kwargs.pop("timestep")
        self.n_iterations: int = int(kwargs.pop("n_iterations"))
        self.integrator: str = kwargs.pop("integrator")
        self.method: str = kwargs.pop("method")
        self.atol: float = kwargs.pop("atol")
        self.rtol: float = kwargs.pop("rtol")
        self.max_step: float = kwargs.pop("max_step")
        self.checks: bool = kwargs.pop("checks")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def defaults(cls, **kwargs):

        options = {}

        options["timestep"] = kwargs.pop("timestep", 1e-3)
        options["n_iterations"] = int(kwargs.pop("n_iterations", 1e3))
        options["integrator"] = kwargs.pop("integrator", "rk4")
        options["method"] = kwargs.pop("method", "adams")
        options["atol"] = kwargs.pop("atol", 1e-12)
        options["rtol"] = kwargs.pop("rtol", 1e-6)
        options["max_step"] = kwargs.pop("max_step", 0.0)
        options["checks"] = kwargs.pop("checks", True)
        return cls(**options)

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        return cls(**kwargs)


###################
# Logging Options #
###################
class NetworkLogOptions(Options):
    """ Log options for the network level

    Configure logging for network events and iterations.

    Attributes:
        n_iterations (int): Number of iterations to log.
        buffer_size (int): Size of the log buffer. Defaults to n_iterations if 0.
        nodes_all (bool): Whether to log all nodes or only selected ones. Defaults to False.
    """

    def __init__(self, n_iterations: int, **kwargs):
        super().__init__(**kwargs)

        self.n_iterations: int = n_iterations
        assert isinstance(self.n_iterations, int), "iterations shoulde be an integer"
        self.buffer_size: int = kwargs.pop('buffer_size', self.n_iterations)
        if self.buffer_size == 0:
            self.buffer_size = self.n_iterations
        assert isinstance(self.buffer_size, int), "buffer_size shoulde be an integer"
        self.nodes_all: bool = kwargs.pop("nodes_all", False)

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """ From options """
        n_iterations = kwargs.pop("n_iterations")
        return cls(n_iterations, **kwargs)
