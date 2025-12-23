"""
Common motion and measurement models for state estimation.

This module provides reusable motion models (process models) and measurement models
used across multiple estimators and examples.

Benefits:
- Code reuse and consistency
- Tested implementations
- Clear documentation
- Easy to extend
"""

from .motion_models import (
    ConstantVelocity2D,
    ConstantVelocity1D,
    ConstantAcceleration2D,
    create_process_noise_continuous_white_acceleration
)

from .measurement_models import (
    RangeMeasurement2D,
    RangeBearingMeasurement2D,
    PositionMeasurement2D,
    validate_measurement_inputs
)

__all__ = [
    # Motion models
    'ConstantVelocity2D',
    'ConstantVelocity1D',
    'ConstantAcceleration2D',
    'create_process_noise_continuous_white_acceleration',
    
    # Measurement models
    'RangeMeasurement2D',
    'RangeBearingMeasurement2D',
    'PositionMeasurement2D',
    'validate_measurement_inputs',
]

