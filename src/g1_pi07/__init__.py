"""Unitree G1 43-DoF data, training, and deployment extensions.

The underlying PaliGemma and Flow Matching model code remains in
:mod:`openpi` and retains its upstream attribution.
"""

from g1_pi07.joints import ACTIVE_ACTION_DIM
from g1_pi07.joints import FULL_DOF
from g1_pi07.joints import MODEL_ACTION_DIM

__all__ = ["ACTIVE_ACTION_DIM", "FULL_DOF", "MODEL_ACTION_DIM"]

__version__ = "0.1.0"
