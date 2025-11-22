"""
FADE: Fuzzy Associative Degradation Engine

A novel AI memory architecture that uses deliberate forgetting
as a confidence signal for retrieval-augmented generation.
"""

__version__ = "0.1.0"

from .model import TinyTransformer
from .strength import StrengthTracker
from .degradation import DegradationModule
from .fuzziness import FuzzinessDetector
from .fade_model import FADEModel

__all__ = [
    "TinyTransformer",
    "StrengthTracker",
    "DegradationModule",
    "FuzzinessDetector",
    "FADEModel",
]
