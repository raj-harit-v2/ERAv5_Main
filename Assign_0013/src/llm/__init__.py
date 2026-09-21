"""20M decoder and reversible residual rules."""

from src.llm.reversible import euler_step, midpoint_reconstruct, midpoint_step
from src.llm.tiny_lm import LMConfig, TinyReversibleLM, count_parameters

__all__ = [
    "LMConfig",
    "TinyReversibleLM",
    "count_parameters",
    "euler_step",
    "midpoint_step",
    "midpoint_reconstruct",
]
