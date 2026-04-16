# __init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("slot-toolkit")
except PackageNotFoundError:
    __version__ = "unknown"

from .data import norm_data
from .model import SLOT_model
from .plot  import *
from . import go