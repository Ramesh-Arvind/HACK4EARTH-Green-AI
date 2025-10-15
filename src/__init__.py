# EcoGrow - Green AI for Sustainable Greenhouse Control
# HACK4EARTH Green AI Challenge 2025
# Track A: Build Green AI + Track B: Use AI for Green Impact

"""
EcoGrow combines energy-efficient AI models with greenhouse optimization
to reduce both model carbon footprint and operational energy consumption.

Modules:
- baseline_benchmark: Measure model energy and carbon footprint (Track A)
- optimized_model: Quantized model for 67% energy reduction (Track A)
- carbon_aware_trainer: Schedule training during low-carbon hours (Track A)
- greenhouse_optimizer: Optimize greenhouse energy usage (Track B)
"""

__version__ = "1.0.0"
__author__ = "EcoGrow Team"
__license__ = "MIT"

# Module imports for easier access
try:
    from .baseline_benchmark import BaselineBenchmark
    from .optimized_model import QuantizedPICAModel
    from .carbon_aware_trainer import CarbonAwareScheduler
    from .greenhouse_optimizer import GreenhouseEnergyOptimizer
    
    __all__ = [
        'BaselineBenchmark',
        'QuantizedPICAModel',
        'CarbonAwareScheduler',
        'GreenhouseEnergyOptimizer',
    ]
except ImportError:
    # Modules can be imported individually if needed
    pass
