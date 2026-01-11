"""
Adversarial package for spam detection
"""
from .generator import AdversarialGenerator
from .hardening import ModelHardening

__all__ = ['AdversarialGenerator', 'ModelHardening']
