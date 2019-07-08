import sys
from pip._internal.operations import freeze
from logging import getLogger

logger = getLogger(__name__)


def log_versions():
    logger.info(sys.version)  # Python version
    logger.info(','.join(freeze.freeze()))  # pip freeze
