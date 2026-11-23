"""dlrand - exact, reproducible random variate generation (C engines + Python interface).

The normal distribution is called the DL (De Moivre-Laplace) distribution throughout this library.
"""
from ._lib import EnvelopeError, DlrandError, RejectionTimeoutError, load_library
from .generator import Proposal, RandomGenerator
from .mixture import AliasSampler, Mixture, MixtureBuilder
from .optimizer import Analysis, ProposalPlan, RejectionOptimizer

__version__ = "0.1.0"
__all__ = ["RandomGenerator", "Proposal", "MixtureBuilder", "Mixture", "AliasSampler", "RejectionOptimizer",
           "ProposalPlan", "Analysis", "DlrandError", "RejectionTimeoutError", "EnvelopeError", "load_library",
           "__version__"]
