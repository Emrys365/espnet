import pathlib
from .evaluation import InputMetrics, OutputMetrics

project_root = pathlib.Path(__file__).expanduser().absolute().parent.parent
__version__ = '0.0.2'
