"""
Data structures for proprioceptive and environmental sensors (Chapter 6).

This module defines the shared data types used across all Chapter 6 algorithms:
    - Frame convention definitions (coordinate systems and quaternion conventions)
    - Sensor time-series packets (IMU, wheel speed, magnetometer, barometer)
    - Navigation state representations (quaternion-velocity-position)
    - Bias-augmented state for realistic INS/EKF implementations

All structures use NumPy arrays for efficient numerical operations and are
designed to work seamlessly with core/estimators and core/sim modules.

Time Base Convention:
    All timestamps are float seconds (monotonic), stored as np.ndarray.

Frame Conventions:
    - B: Body frame (sensor frame)
    - M: Map frame (navigation frame, ENU by default)
    - S: Speed frame (for wheel odometry, defined in Chapter 6)
    - A: Attitude frame (intermediate frame for vehicle odometry)

See FrameConvention dataclass for explicit coordinate system definitions.

References:
    Chapter 6: Dead Reckoning and Proprioceptive Sensors
    Section 6.1: IMU strapdown integration
    Section 6.2: Wheel odometry
    Section 6.3: Pedestrian dead reckoning
"""

from dataclasses import dataclass
from typing import Dict, Any, Literal
import numpy as np


@dataclass(frozen=True)
class FrameConvention:
    """
    Explicit coordinate frame and orientation conventions for Chapter 6.

    This dataclass defines the reference frames and conventions used throughout
    all Chapter 6 dead reckoning algorithms. Unifies frame definitions across:
        - IMU strapdown integration
        - Wheel odometry
        - Pedestrian dead reckoning (PDR)
        - Magnetometer heading estimation
        - All sensor fusion methods

    Attributes:
        map_frame: Name of the map/navigation frame.
                   Default: 'ENU' (East-North-Up).
                   Options: 'ENU', 'NED' (North-East-Down).
        map_axes: Axis definitions in map frame.
                  For 'ENU': ('x=East', 'y=North', 'z=Up')
                  For 'NED': ('x=North', 'y=East', 'z=Down')
        gravity_direction: Direction of gravity vector in map frame.
                           For 'ENU': -1 (downward, negative z)
                           For 'NED': +1 (downward, positive z)
        heading_zero_direction: Direction corresponding to heading = 0.
                                For 'ENU': 'East' (0 rad = +x axis)
                                For 'NED': 'North' (0 rad = +x axis)
        heading_increases_towards: Direction of increasing heading angle.
                                   For 'ENU': 'North' (counter-clockwise from East)
                                   For 'NED': 'East' (clockwise from North)
        quaternion_convention: Quaternion component order.
                               Default: 'scalar_first' ([q0, q1, q2, q3])
                               Options: 'scalar_first', 'scalar_last'
        quaternion_meaning: What the quaternion represents.
                            Default: 'body_to_map' (C_B^M rotation)
                            Options: 'body_to_map', 'map_to_body'
        body_frame_axes: Body frame axis definitions (IMU frame).
                         Default: ('x=forward', 'y=left', 'z=up')
                         Standard for forward-facing device/robot.

    Notes:
        - All Chapter 6 algorithms should use this convention explicitly.
        - Default is ENU with heading 0 = East (consistent with Eq. 6.50).
        - Quaternion is body-to-map (v_map = C(q) @ v_body).
        - Gravity vector: g_map = [0, 0, gravity_direction * 9.81] (Eq. 6.8).
        - Heading convention (Eq. 6.50): p_k = p_{k-1} + L*[cos(ψ), sin(ψ)]
          implies ψ=0 → +x (East), ψ=π/2 → +y (North) for ENU.

    Example:
        >>> # Default ENU convention
        >>> frame = FrameConvention()
        >>> print(frame.map_frame)  # 'ENU'
        >>> print(frame.heading_zero_direction)  # 'East'
        >>> print(frame.gravity_direction)  # -1 (downward in ENU)
        >>>
        >>> # Create NED convention
        >>> frame_ned = FrameConvention.create_ned()
        >>> print(frame_ned.map_frame)  # 'NED'
        >>> print(frame_ned.gravity_direction)  # +1 (downward in NED)

    Related Equations:
        - Eq. (6.8): Gravity vector definition
        - Eq. (6.50): PDR position update (defines heading convention)
        - Eq. (6.52)-(6.53): Magnetometer heading (must match frame)
    """

    map_frame: Literal['ENU', 'NED'] = 'ENU'
    map_axes: tuple[str, str, str] = ('x=East', 'y=North', 'z=Up')
    gravity_direction: Literal[-1, +1] = -1
    heading_zero_direction: Literal['East', 'North'] = 'East'
    heading_increases_towards: Literal['North', 'East'] = 'North'
    quaternion_convention: Literal['scalar_first', 'scalar_last'] = 'scalar_first'
    quaternion_meaning: Literal['body_to_map', 'map_to_body'] = 'body_to_map'
    body_frame_axes: tuple[str, str, str] = (
        'x=forward',
        'y=left',
        'z=up',
    )

    def __post_init__(self) -> None:
        """Validate frame convention consistency."""
        # Validate map frame matches axes and gravity
        if self.map_frame == 'ENU':
            expected_axes = ('x=East', 'y=North', 'z=Up')
            expected_gravity = -1
            expected_heading_zero = 'East'
            expected_heading_increases = 'North'
        elif self.map_frame == 'NED':
            expected_axes = ('x=North', 'y=East', 'z=Down')
            expected_gravity = +1
            expected_heading_zero = 'North'
            expected_heading_increases = 'East'
        else:
            raise ValueError(
                f"map_frame must be 'ENU' or 'NED', got '{self.map_frame}'"
            )

        if self.map_axes != expected_axes:
            raise ValueError(
                f"For {self.map_frame}, map_axes should be {expected_axes}, "
                f"got {self.map_axes}"
            )

        if self.gravity_direction != expected_gravity:
            raise ValueError(
                f"For {self.map_frame}, gravity_direction should be "
                f"{expected_gravity}, got {self.gravity_direction}"
            )

        if self.heading_zero_direction != expected_heading_zero:
            raise ValueError(
                f"For {self.map_frame}, heading_zero_direction should be "
                f"'{expected_heading_zero}', got '{self.heading_zero_direction}'"
            )

        if self.heading_increases_towards != expected_heading_increases:
            raise ValueError(
                f"For {self.map_frame}, heading_increases_towards should be "
                f"'{expected_heading_increases}', "
                f"got '{self.heading_increases_towards}'"
            )

    @classmethod
    def create_enu(cls) -> "FrameConvention":
        """
        Create ENU (East-North-Up) frame convention.

        This is the default convention for Chapter 6 examples, consistent with
        Eq. (6.50) PDR update where heading 0 = East.

        Returns:
            FrameConvention with ENU settings.

        Example:
            >>> frame = FrameConvention.create_enu()
            >>> print(frame.map_frame)  # 'ENU'
            >>> print(frame.heading_zero_direction)  # 'East'
        """
        return cls(
            map_frame='ENU',
            map_axes=('x=East', 'y=North', 'z=Up'),
            gravity_direction=-1,
            heading_zero_direction='East',
            heading_increases_towards='North',
        )

    @classmethod
    def create_ned(cls) -> "FrameConvention":
        """
        Create NED (North-East-Down) frame convention.

        Alternative convention used in some aerospace applications.
        Gravity points downward (positive z).

        Returns:
            FrameConvention with NED settings.

        Example:
            >>> frame = FrameConvention.create_ned()
            >>> print(frame.map_frame)  # 'NED'
            >>> print(frame.gravity_direction)  # +1
        """
        return cls(
            map_frame='NED',
            map_axes=('x=North', 'y=East', 'z=Down'),
            gravity_direction=+1,
            heading_zero_direction='North',
            heading_increases_towards='East',
        )

    def gravity_vector(self, g_mag: float = 9.81) -> np.ndarray:
        """
        Get gravity vector in map frame (Eq. 6.8).

        Args:
            g_mag: Gravitational acceleration magnitude.
                   Default: 9.81 m/s².

        Returns:
            Gravity vector in map frame.
            Shape: (3,). Units: m/s².
            For ENU: [0, 0, -9.81] (downward)
            For NED: [0, 0, +9.81] (downward)

        Example:
            >>> frame = FrameConvention.create_enu()
            >>> g = frame.gravity_vector()
            >>> print(g)  # [0, 0, -9.81]
        """
        return np.array([0.0, 0.0, self.gravity_direction * g_mag])

    def heading_to_unit_vector(self, heading_rad: float) -> np.ndarray:
        """
        Convert heading angle to unit direction vector in horizontal plane.

        For ENU (heading 0 = East):
            heading=0   → [1, 0] (East)
            heading=π/2 → [0, 1] (North)

        For NED (heading 0 = North):
            heading=0   → [1, 0] (North)
            heading=π/2 → [0, 1] (East)

        Args:
            heading_rad: Heading angle in radians.

        Returns:
            Unit direction vector in horizontal plane.
            Shape: (2,). Components: [x, y] in map frame.

        Example:
            >>> frame = FrameConvention.create_enu()
            >>> v = frame.heading_to_unit_vector(0.0)  # East
            >>> print(v)  # [1, 0]
            >>> v = frame.heading_to_unit_vector(np.pi/2)  # North
            >>> print(v)  # [0, 1]
        """
        if self.map_frame == 'ENU':
            # ENU: heading 0 = East (+x), π/2 = North (+y)
            return np.array([np.cos(heading_rad), np.sin(heading_rad)])
        else:  # NED
            # NED: heading 0 = North (+x), π/2 = East (+y)
            return np.array([np.cos(heading_rad), np.sin(heading_rad)])


@dataclass(frozen=True)
class IMUNoiseParams:
    """
    IMU noise and bias parameters with explicit units in field names.
    
    This dataclass eliminates ambiguity in IMU specification by making units
    explicit in every field name. All values are stored in SI units (rad/s, m/s²)
    with the original specification units documented.
    
    Key principle: ALWAYS use explicit unit names to prevent deg/hr vs deg/s bugs!
    
    Attributes:
        gyro_bias_rad_s: Gyroscope bias instability (rad/s).
                         Spec sheet unit: deg/hr.
                         Typical values: 0.1-100 deg/hr (consumer),
                                        0.01-1 deg/hr (tactical).
        gyro_arw_rad_sqrt_s: Angular Random Walk coefficient (rad/√s).
                             Spec sheet unit: deg/√hr.
                             Typical values: 0.05-1.0 deg/√hr (consumer),
                                            0.005-0.05 deg/√hr (tactical).
        gyro_rrw_rad_s_sqrt_s: Rate Random Walk coefficient (rad/s/√s).
                               Spec sheet unit: deg/s/√hr.
                               Typical values: 0.001-0.1 deg/s/√hr.
        accel_bias_mps2: Accelerometer bias instability (m/s²).
                         Spec sheet unit: mg (milligravity).
                         Typical values: 1-100 mg (consumer),
                                        0.01-1 mg (tactical).
        accel_vrw_mps_sqrt_s: Velocity Random Walk coefficient (m/s/√s).
                              Spec sheet unit: m/s/√hr.
                              Typical values: 0.001-0.1 m/s/√hr.
        grade: IMU grade ('consumer', 'tactical', 'navigation').
               For documentation purposes only.
    
    Example:
        >>> from core.sensors.units import (
        ...     deg_per_hour_to_rad_per_sec,
        ...     deg_per_sqrt_hour_to_rad_per_sqrt_sec,
        ...     mg_to_mps2,
        ...     mps_per_sqrt_hour_to_mps_per_sqrt_sec
        ... )
        >>> 
        >>> # Consumer-grade IMU (explicit unit conversions)
        >>> params = IMUNoiseParams(
        ...     gyro_bias_rad_s=deg_per_hour_to_rad_per_sec(10.0),
        ...     gyro_arw_rad_sqrt_s=deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.1),
        ...     gyro_rrw_rad_s_sqrt_s=0.0,  # Not specified
        ...     accel_bias_mps2=mg_to_mps2(10.0),
        ...     accel_vrw_mps_sqrt_s=mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01),
        ...     grade='consumer'
        ... )
        >>> print(f"Gyro bias: {params.gyro_bias_rad_s:.6e} rad/s")
        Gyro bias: 4.848137e-05 rad/s
    
    Related Equations:
        - Eq. (6.5): Gyro measurement model (b_G term)
        - Eq. (6.6): Gyro bias correction
        - Eq. (6.9): Accelerometer measurement model (b_A term)
        - Eqs. (6.56)-(6.58): Allan variance analysis
    
    Notes:
        - Use core.sensors.units module for ALL conversions
        - Print diagnostics should use units.format_* functions
        - Never use bare np.deg2rad() for bias conversions!
    """
    
    gyro_bias_rad_s: float
    gyro_arw_rad_sqrt_s: float
    gyro_rrw_rad_s_sqrt_s: float
    accel_bias_mps2: float
    accel_vrw_mps_sqrt_s: float
    grade: str = 'unknown'
    
    @classmethod
    def consumer_grade(cls) -> "IMUNoiseParams":
        """
        Create typical consumer-grade IMU noise parameters.
        
        Based on typical smartphone/tablet IMU specifications.
        
        Returns:
            Consumer-grade IMU parameters.
        
        Example:
            >>> params = IMUNoiseParams.consumer_grade()
            >>> print(params.grade)  # 'consumer'
        """
        from core.sensors.units import (
            deg_per_hour_to_rad_per_sec,
            deg_per_sqrt_hour_to_rad_per_sqrt_sec,
            mg_to_mps2,
            mps_per_sqrt_hour_to_mps_per_sqrt_sec,
        )
        
        return cls(
            gyro_bias_rad_s=deg_per_hour_to_rad_per_sec(10.0),  # 10 deg/hr
            gyro_arw_rad_sqrt_s=deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.1),  # 0.1 deg/√hr
            gyro_rrw_rad_s_sqrt_s=0.0,
            accel_bias_mps2=mg_to_mps2(10.0),  # 10 mg
            accel_vrw_mps_sqrt_s=mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01),  # 0.01 m/s/√hr
            grade='consumer',
        )
    
    @classmethod
    def tactical_grade(cls) -> "IMUNoiseParams":
        """
        Create typical tactical-grade IMU noise parameters.
        
        Based on mid-range fiber optic gyro (FOG) or MEMS specifications.
        
        Returns:
            Tactical-grade IMU parameters.
        """
        from core.sensors.units import (
            deg_per_hour_to_rad_per_sec,
            deg_per_sqrt_hour_to_rad_per_sqrt_sec,
            mg_to_mps2,
            mps_per_sqrt_hour_to_mps_per_sqrt_sec,
        )
        
        return cls(
            gyro_bias_rad_s=deg_per_hour_to_rad_per_sec(1.0),  # 1 deg/hr
            gyro_arw_rad_sqrt_s=deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.01),  # 0.01 deg/√hr
            gyro_rrw_rad_s_sqrt_s=0.0,
            accel_bias_mps2=mg_to_mps2(1.0),  # 1 mg
            accel_vrw_mps_sqrt_s=mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.001),  # 0.001 m/s/√hr
            grade='tactical',
        )
    
    @classmethod
    def navigation_grade(cls) -> "IMUNoiseParams":
        """
        Create typical navigation-grade IMU noise parameters.
        
        Based on ring laser gyro (RLG) specifications.
        
        Returns:
            Navigation-grade IMU parameters.
        """
        from core.sensors.units import (
            deg_per_hour_to_rad_per_sec,
            deg_per_sqrt_hour_to_rad_per_sqrt_sec,
            mg_to_mps2,
            mps_per_sqrt_hour_to_mps_per_sqrt_sec,
        )
        
        return cls(
            gyro_bias_rad_s=deg_per_hour_to_rad_per_sec(0.01),  # 0.01 deg/hr
            gyro_arw_rad_sqrt_s=deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.001),  # 0.001 deg/√hr
            gyro_rrw_rad_s_sqrt_s=0.0,
            accel_bias_mps2=mg_to_mps2(0.1),  # 0.1 mg
            accel_vrw_mps_sqrt_s=mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.0001),  # 0.0001 m/s/√hr
            grade='navigation',
        )
    
    def format_specs(self) -> str:
        """
        Format IMU specifications for human-readable display.
        
        Returns:
            Multi-line formatted string with all parameters.
        
        Example:
            >>> params = IMUNoiseParams.consumer_grade()
            >>> print(params.format_specs())
            IMU Specifications (consumer grade):
              Gyro Bias:  10.00 deg/hr (0.0028 deg/s)
              Gyro ARW:   0.10 deg/√hr
              Accel Bias: 10.00 mg (0.0981 m/s²)
              Accel VRW:  0.0100 m/s/√hr
        """
        from core.sensors.units import (
            format_gyro_bias,
            format_arw,
            format_accel_bias,
            format_vrw,
        )
        
        lines = [
            f"IMU Specifications ({self.grade} grade):",
            f"  Gyro Bias:  {format_gyro_bias(self.gyro_bias_rad_s)}",
            f"  Gyro ARW:   {format_arw(self.gyro_arw_rad_sqrt_s)}",
            f"  Accel Bias: {format_accel_bias(self.accel_bias_mps2)}",
            f"  Accel VRW:  {format_vrw(self.accel_vrw_mps_sqrt_s)}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ImuSeries:
    """
    Time-series packet for Inertial Measurement Unit (IMU) data.

    Stores synchronized accelerometer and gyroscope measurements in the
    body frame B, used for strapdown integration (Eqs. (6.2)-(6.10)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        accel: Specific force measurements in body frame B, shape (N, 3).
               Units: m/s². Includes gravity (before correction via Eq. (6.9)).
        gyro: Angular velocity measurements in body frame B, shape (N, 3).
              Units: rad/s. Raw measurements (before correction via Eq. (6.6)).
        meta: Optional metadata dict. May include:
              - 'sample_rate_hz': float, nominal sampling rate
              - 'sensor_id': str, device identifier
              - 'frame': str, typically 'body' or 'B'
              - 'units': dict, e.g. {'accel': 'm/s^2', 'gyro': 'rad/s'}

    Notes:
        - Measurements are assumed to be in the body frame at each timestamp.
        - For error modeling with biases, see NavStateQPVPBias and Eqs. (6.5)-(6.6).
        - frozen=True ensures immutability for safer data pipelines.

    Related Equations:
        - Eq. (6.6): Gyro correction (ω = ω̃ - b_g - n_g)
        - Eq. (6.9): Accel correction (f = f̃ - b_a - n_a)
        - Eq. (6.2): Quaternion kinematics using corrected gyro
        - Eq. (6.7): Velocity update using corrected accel
    """

    t: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency of IMU data."""
        # Validate t is 1D
        if self.t.ndim != 1:
            raise ValueError(
                f"ImuSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        # Validate accel is (N, 3)
        if self.accel.shape != (n_samples, 3):
            raise ValueError(
                f"ImuSeries.accel must have shape ({n_samples}, 3), "
                f"got {self.accel.shape}"
            )

        # Validate gyro is (N, 3)
        if self.gyro.shape != (n_samples, 3):
            raise ValueError(
                f"ImuSeries.gyro must have shape ({n_samples}, 3), "
                f"got {self.gyro.shape}"
            )


@dataclass(frozen=True)
class WheelSpeedSeries:
    """
    Time-series packet for wheel speed / vehicle odometry data.

    Stores velocity measurements in the speed frame S (defined in Chapter 6
    for wheel odometry systems). Used for vehicle dead reckoning and
    integrated IMU+wheel EKF (Eqs. (6.11)-(6.15), (6.33)-(6.38)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        v_s: Velocity in speed frame S, shape (N, 3). Units: m/s.
             Typically v_s = [v_x, 0, 0] for forward motion in a vehicle.
             See Eq. (6.11) for the speed frame definition.
        meta: Optional metadata dict. May include:
              - 'lever_arm_b': np.ndarray (3,), lever arm in body frame (meters)
              - 'frame': str, typically 'speed' or 'S'
              - 'vehicle_type': str, e.g. 'differential_drive', 'ackermann'

    Notes:
        - Speed frame S is typically aligned with vehicle forward direction.
        - Lever arm compensation (Eq. (6.11)) transforms v_s to attitude frame.
        - For slip/failure mode demos, inject noise or bias in v_s.

    Related Equations:
        - Eq. (6.11): Lever arm compensation (v^A = v^S - [ω_B ×] l^B)
        - Eq. (6.14): Attitude to map velocity transform
        - Eq. (6.15): Position update using wheel speed
        - Eqs. (6.33)-(6.38): Wheel speed measurement model for EKF
    """

    t: np.ndarray
    v_s: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency of wheel speed data."""
        if self.t.ndim != 1:
            raise ValueError(
                f"WheelSpeedSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        if self.v_s.shape != (n_samples, 3):
            raise ValueError(
                f"WheelSpeedSeries.v_s must have shape ({n_samples}, 3), "
                f"got {self.v_s.shape}"
            )


@dataclass(frozen=True)
class MagnetometerSeries:
    """
    Time-series packet for magnetometer data.

    Stores magnetic field measurements in the device/body frame, used for
    heading estimation in PDR and environmental sensor fusion
    (Eqs. (6.51)-(6.53)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        mag: Magnetic field vector in body/device frame, shape (N, 3).
             Units: μT (microtesla) or normalized. Typically contains
             Earth's magnetic field plus indoor disturbances.
        meta: Optional metadata dict. May include:
              - 'frame': str, typically 'body' or 'device'
              - 'calibration': dict, hard-iron/soft-iron calibration params
              - 'disturbance_intervals': list of (start_idx, end_idx) tuples
              - 'units': str, e.g. 'uT', 'normalized'

    Notes:
        - Indoor environments often have magnetic disturbances (steel, electronics).
        - Tilt compensation (Eq. (6.52)) requires attitude (roll/pitch) from IMU.
        - Raw measurements must be calibrated for hard-iron and soft-iron effects.

    Related Equations:
        - Eq. (6.51): Magnetometer heading definition
        - Eq. (6.52): Tilt compensation for heading
        - Eq. (6.53): Heading computation from tilt-compensated mag field
    """

    t: np.ndarray
    mag: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency of magnetometer data."""
        if self.t.ndim != 1:
            raise ValueError(
                f"MagnetometerSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        if self.mag.shape != (n_samples, 3):
            raise ValueError(
                f"MagnetometerSeries.mag must have shape ({n_samples}, 3), "
                f"got {self.mag.shape}"
            )


@dataclass(frozen=True)
class BarometerSeries:
    """
    Time-series packet for barometric pressure data.

    Stores atmospheric pressure measurements for altitude estimation and
    floor change detection (Eq. (6.54)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        pressure: Atmospheric pressure measurements, shape (N,).
                  Units: Pa (Pascals) or hPa (hectopascals).
                  Must be explicit in meta['units'].
        meta: Optional metadata dict. May include:
              - 'units': str, REQUIRED ('Pa' or 'hPa')
              - 'p0': float, reference pressure at sea level (Pa)
              - 'T': float, temperature in Kelvin for altitude model
              - 'floor_labels': np.ndarray (N,), manual floor annotations

    Notes:
        - Pressure-to-altitude conversion via Eq. (6.54) requires p0 and T.
        - Barometer drift and offset handling are critical for long-term use.
        - Indoor pressure can be affected by HVAC systems and weather changes.

    Related Equations:
        - Eq. (6.54): Pressure to altitude conversion (barometric formula)
        - Eq. (6.55): Generic state/measurement model (for smoothing helper)
    """

    t: np.ndarray
    pressure: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency and units specification."""
        if self.t.ndim != 1:
            raise ValueError(
                f"BarometerSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        if self.pressure.shape != (n_samples,):
            raise ValueError(
                f"BarometerSeries.pressure must have shape ({n_samples},), "
                f"got {self.pressure.shape}"
            )

        # Enforce units specification
        if "units" not in self.meta:
            raise ValueError(
                "BarometerSeries.meta must include 'units' field ('Pa' or 'hPa')"
            )


@dataclass
class NavStateQPVP:
    """
    Minimal navigation state: Quaternion, Velocity, Position (QPVP).

    Represents the core navigation state used in Chapter 6 strapdown
    integration and EKF formulations (base state for Eq. (6.16)).

    Attributes:
        q: Quaternion representing attitude (body to map frame rotation).
           Shape (4,), scalar-first convention: [q0, q1, q2, q3] where
           q0 is the scalar part and [q1, q2, q3] is the vector part.
           Must satisfy ||q|| = 1 (unit quaternion).
        v: Velocity in map frame M, shape (3,). Units: m/s.
           Typically M is ENU (East-North-Up) or NED (North-East-Down).
        p: Position in map frame M, shape (3,). Units: m.
           3D position [x, y, z] or [E, N, U] depending on frame choice.

    Notes:
        - This is a MUTABLE dataclass (frozen=False) to allow state updates.
        - Quaternion normalization is the user's responsibility after updates.
        - Use this for simple examples; use NavStateQPVPBias for realistic EKF.

    Related Equations:
        - Eq. (6.2): Quaternion kinematics (dq/dt = 0.5 * Ω(ω) * q)
        - Eq. (6.7): Velocity update (v_k = v_{k-1} + (C_B^M f + g) Δt)
        - Eq. (6.10): Position update (p_k = p_{k-1} + v_k Δt)
        - Eq. (6.16): State definition for IMU+wheel EKF (with biases)

    Example:
        >>> import numpy as np
        >>> # Initial state: zero velocity, origin position, level attitude
        >>> q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        >>> v0 = np.zeros(3)
        >>> p0 = np.zeros(3)
        >>> state = NavStateQPVP(q=q0, v=v0, p=p0)
    """

    q: np.ndarray
    v: np.ndarray
    p: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape and basic consistency of navigation state."""
        if self.q.shape != (4,):
            raise ValueError(
                f"NavStateQPVP.q must have shape (4,), got {self.q.shape}"
            )

        if self.v.shape != (3,):
            raise ValueError(
                f"NavStateQPVP.v must have shape (3,), got {self.v.shape}"
            )

        if self.p.shape != (3,):
            raise ValueError(
                f"NavStateQPVP.p must have shape (3,), got {self.p.shape}"
            )

        # Warn if quaternion is not normalized (tolerance 1e-3)
        q_norm = np.linalg.norm(self.q)
        if not np.isclose(q_norm, 1.0, atol=1e-3):
            import warnings

            warnings.warn(
                f"NavStateQPVP initialized with non-unit quaternion "
                f"(||q|| = {q_norm:.6f}). Consider normalizing.",
                UserWarning,
            )


@dataclass
class NavStateQPVPBias:
    """
    Augmented navigation state: Quaternion, Velocity, Position, and Biases.

    Extended state including IMU biases for realistic EKF implementations.
    Recommended for Chapter 6 EKF examples that model sensor errors
    (Eqs. (6.5)-(6.6), (6.9), and state definition in Eq. (6.16)).

    Attributes:
        q: Quaternion (body to map frame), shape (4,). Scalar-first [q0, q1, q2, q3].
        v: Velocity in map frame M, shape (3,). Units: m/s.
        p: Position in map frame M, shape (3,). Units: m.
        b_g: Gyroscope bias in body frame B, shape (3,). Units: rad/s.
             Slow-varying bias modeled as random walk (Eq. (6.5)).
        b_a: Accelerometer bias in body frame B, shape (3,). Units: m/s².
             Slow-varying bias modeled as random walk (Eq. (6.9)).

    Notes:
        - Full state dimension is 16 (3 + 3 + 4 + 3 + 3).
        - Bias evolution is typically modeled as random walk in EKF process model.
        - For error-state EKF, biases are estimated as additive corrections.

    Related Equations:
        - Eq. (6.5): Gyro error model (ω̃ = ω + b_g + n_g)
        - Eq. (6.6): Gyro correction (ω = ω̃ - b_g - n_g)
        - Eq. (6.9): Accel error model (f̃ = f + b_a + n_a)
        - Eq. (6.16): State vector for IMU+wheel EKF [p, v, q, b_g, b_a]
        - Eqs. (6.17)-(6.32): Process model and covariance propagation

    Example:
        >>> import numpy as np
        >>> # Initial state with small biases
        >>> q0 = np.array([1.0, 0.0, 0.0, 0.0])
        >>> v0 = np.zeros(3)
        >>> p0 = np.zeros(3)
        >>> b_g0 = np.array([1e-4, -5e-5, 2e-5])  # rad/s
        >>> b_a0 = np.array([0.01, -0.005, 0.02])  # m/s^2
        >>> state = NavStateQPVPBias(q=q0, v=v0, p=p0, b_g=b_g0, b_a=b_a0)
    """

    q: np.ndarray
    v: np.ndarray
    p: np.ndarray
    b_g: np.ndarray
    b_a: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape consistency of augmented state."""
        if self.q.shape != (4,):
            raise ValueError(
                f"NavStateQPVPBias.q must have shape (4,), got {self.q.shape}"
            )

        if self.v.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.v must have shape (3,), got {self.v.shape}"
            )

        if self.p.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.p must have shape (3,), got {self.p.shape}"
            )

        if self.b_g.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.b_g must have shape (3,), got {self.b_g.shape}"
            )

        if self.b_a.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.b_a must have shape (3,), got {self.b_a.shape}"
            )

        # Warn if quaternion is not normalized
        q_norm = np.linalg.norm(self.q)
        if not np.isclose(q_norm, 1.0, atol=1e-3):
            import warnings

            warnings.warn(
                f"NavStateQPVPBias initialized with non-unit quaternion "
                f"(||q|| = {q_norm:.6f}). Consider normalizing.",
                UserWarning,
            )


