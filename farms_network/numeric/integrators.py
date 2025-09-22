""" Integrators """


class RK4:
    """ RK4 Integrator """

    def __init__(self, system, integration_options):
        " Integration "
        self._rk4_integrator = None
