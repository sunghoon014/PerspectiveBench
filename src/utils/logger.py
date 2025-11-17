import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger as _logger

load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PRINT_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def define_log_level(logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level."""
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # name a log with prefix name

    _logger.remove()
    _logger.add(sys.stderr, level=PRINT_LEVEL)
    _logger.add(f"{PROJECT_ROOT}/resources/logs/{log_name}.log", level=logfile_level)
    return _logger


logger = define_log_level()
