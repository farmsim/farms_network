""" Log module init including LOGGER and logstr """

from .log import Logger

LOGGER = Logger()
debug = LOGGER.debug
info = LOGGER.info
warning = LOGGER.warning
error = LOGGER.error
critical = LOGGER.critical
exception = LOGGER.exception


def set_name(name):
    """ Set logger name """
    LOGGER.name = name
    return
