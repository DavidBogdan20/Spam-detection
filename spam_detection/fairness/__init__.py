"""
Fairness package for spam detection
"""
from .metrics import FairnessMetrics
from .mitigation import BiasMitigation

__all__ = ['FairnessMetrics', 'BiasMitigation']
