from abc import ABC
from enum import Enum, unique
from typing import Type, Union


class BaseTypes(Enum):
    """ Base class for enum types"""

    @classmethod
    def to_str(cls, value: Union[str, Type]) -> Type:
        if isinstance(value, cls):
            return value.value
        if value in cls._value2member_map_:
            return value
        valid_types = ", ".join(type.value for type in cls)
        raise ValueError(f"Invalid type '{value}'. Must be one of: {valid_types}")


@unique
class Models(str, BaseTypes):
    BASE = "base"
    RELAY = "relay"
    LINEAR = "linear"
    RELU = "relu"
    OSCILLATOR = "oscillator"
    HOPF_OSCILLATOR = "hopf_oscillator"
    MORPHED_OSCILLATOR = "morphed_oscillator"
    MATSUOKA = "matsuoka"
    FITZHUGH_NAGUMO = "fitzhugh_nagumo"
    MORRIS_LECAR = "morris_lecar"
    LEAKY_INTEGRATOR = "leaky_integrator"
    LI_DANNER = "li_danner"
    LI_NAP_DANNER = "li_nap_danner"
    LI_DAUN = "li_daun"
    HH_DAUN = "hh_daun"


@unique
class EdgeTypes(str, BaseTypes):
    GENERIC = "generic"
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    CHOLINERGIC = "cholinergic"
    PHASE_COUPLING = "phase_coupling"
