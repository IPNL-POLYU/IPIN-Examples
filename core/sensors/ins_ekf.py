"""
EKF-based INS with ZUPT for foot-mounted navigation (Chapter 6).

This module implements a simplified Extended Kalman Filter for Inertial Navigation
System (INS) with Zero-Velocity Update (ZUPT) pseudo-measurements.

State Vector (Eq. 6.16):
    x = [p (3), v (3), q (4), b_g (3), b_a (3)]^T
    Total: 16 states
    
    Where:
        p: Position in map frame [px, py, pz] (m)
        v: Velocity in map frame [vx, vy, vz] (m/s)
        q: Quaternion (body-to-map rotation), scalar-first [q0, q1, q2, q3]
        b_g: Gyroscope bias in body frame [bgx, bgy, bgz] (rad/s)
        b_a: Accelerometer bias in body frame [bax, bay, baz] (m/s²)

Kalman Filter Equations (Eqs. 6.40-6.43):
    Prediction:
        x_k|k-1 = f(x_k-1, u_k-1)  # Strapdown mechanization
        P_k|k-1 = F * P_k-1 * F^T + Q  # Covariance propagation
    
    Update:
        S_k = H * P_k|k-1 * H^T + R  # Innovation covariance (Eq. 6.40)
        K_k = P_k|k-1 * H^T * S_k^(-1)  # Kalman gain (Eq. 6.41)
        x_k|k = x_k|k-1 + K_k * [z_k - h(x_k)]  # State update (Eq. 6.42)
        P_k|k = P_k|k-1 - K_k * H * P_k|k-1  # Covariance update (Eq. 6.43)

ZUPT Measurement (Eq. 6.45):
    z_k = [0, 0, 0]^T  # Zero velocity measurement
    h(x) = v  # Extract velocity from state
    H = [0_3x3, I_3, 0_3x4, 0_3x3, 0_3x3]  # Measurement Jacobian

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.sensors.strapdown import strapdown_update
from core.sensors.types import FrameConvention, IMUNoiseParams
from core.sensors.gravity import gravity_magnitude


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit norm."""
    return q / np.linalg.norm(q)


@dataclass
class INSState:
    """
    INS state vector for EKF-based navigation.
    
    State ordering follows Eq. (6.16): x = [p, v, q, b_g, b_a]^T
    
    Attributes:
        p: Position in map frame (m), shape (3,).
        v: Velocity in map frame (m/s), shape (3,).
        q: Quaternion (body-to-map), shape (4,), scalar-first.
        b_g: Gyroscope bias in body frame (rad/s), shape (3,).
        b_a: Accelerometer bias in body frame (m/s²), shape (3,).
        P: State covariance matrix, shape (16, 16).
    
    Note:
        The state vector is 16-dimensional (not 13 or 15):
        - Position: 3 elements (indices 0:3)
        - Velocity: 3 elements (indices 3:6)
        - Quaternion: 4 elements (indices 6:10)
        - Gyro bias: 3 elements (indices 10:13)
        - Accel bias: 3 elements (indices 13:16)
    """
    p: np.ndarray  # (3,)
    v: np.ndarray  # (3,)
    q: np.ndarray  # (4,)
    b_g: np.ndarray  # (3,)
    b_a: np.ndarray  # (3,)
    P: np.ndarray  # (16, 16)
    
    def to_vector(self) -> np.ndarray:
        """
        Convert state to vector form following Eq. (6.16).
        
        Returns:
            State vector x = [p, v, q, b_g, b_a]^T, shape (16,).
        """
        return np.concatenate([self.p, self.v, self.q, self.b_g, self.b_a])
    
    @classmethod
    def from_vector(cls, x: np.ndarray, P: np.ndarray) -> "INSState":
        """
        Create state from vector form following Eq. (6.16).
        
        Args:
            x: State vector [p, v, q, b_g, b_a]^T, shape (16,).
            P: Covariance matrix, shape (16, 16).
        
        Returns:
            INSState instance.
        """
        if x.shape[0] != 16:
            raise ValueError(f"State vector must have 16 elements, got {x.shape[0]}")
        if P.shape != (16, 16):
            raise ValueError(f"Covariance must be (16, 16), got {P.shape}")
        
        return cls(
            p=x[0:3],
            v=x[3:6],
            q=x[6:10],
            b_g=x[10:13],
            b_a=x[13:16],
            P=P
        )


class ZUPT_EKF:
    """
    Extended Kalman Filter for ZUPT-aided INS.
    
    Implements Equations 6.40-6.43 for Kalman update and Eq. 6.45 for ZUPT.
    
    Args:
        frame: Frame convention (ENU or NED).
        imu_params: IMU noise parameters.
        sigma_zupt: ZUPT measurement noise std dev (m/s). Default: 0.01.
        g: Gravity magnitude (fallback when lat_rad=None, m/s²). Default: 9.81.
        lat_rad: Geodetic latitude in radians (optional).
                 If provided, uses Eq. (6.8) for gravity magnitude.
                 If None, uses g parameter (backward compatible).
    """
    
    def __init__(
        self,
        frame: FrameConvention,
        imu_params: IMUNoiseParams,
        sigma_zupt: float = 0.01,
        g: float = 9.81,
        lat_rad: Optional[float] = None,
    ):
        self.frame = frame
        self.imu_params = imu_params
        self.sigma_zupt = sigma_zupt
        # Compute gravity magnitude using Eq. (6.8) if latitude provided
        self.g = gravity_magnitude(lat_rad=lat_rad, default_g=g)
        self.lat_rad = lat_rad
        
        # Process noise (simplified)
        self.Q = np.zeros((16, 16))
        
    def initialize(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        q0: np.ndarray,
        P0: Optional[np.ndarray] = None
    ) -> INSState:
        """
        Initialize EKF state following Eq. (6.16) ordering.
        
        Args:
            p0: Initial position, shape (3,).
            v0: Initial velocity, shape (3,).
            q0: Initial quaternion, shape (4,).
            P0: Initial covariance, shape (16, 16). If None, uses default.
        
        Returns:
            Initial INSState.
        """
        # Initialize biases to zero
        b_g0 = np.zeros(3)
        b_a0 = np.zeros(3)
        
        # Default covariance if not provided
        if P0 is None:
            P0 = np.eye(16) * 1e-6  # Small initial uncertainty
            # Position uncertainty (indices 0:3)
            P0[0:3, 0:3] *= 0.1**2
            # Velocity uncertainty (indices 3:6)
            P0[3:6, 3:6] *= 0.1**2
            # Attitude uncertainty (indices 6:10, quaternion is normalized constraint)
            P0[6:10, 6:10] *= 1e-4
            # Gyro bias uncertainty (indices 10:13)
            P0[10:13, 10:13] *= self.imu_params.gyro_bias_rad_s**2
            # Accel bias uncertainty (indices 13:16)
            P0[13:16, 13:16] *= self.imu_params.accel_bias_mps2**2
        
        return INSState(p=p0, v=v0, q=q0, b_g=b_g0, b_a=b_a0, P=P0)
    
    def predict(
        self,
        state: INSState,
        gyro_meas: np.ndarray,
        accel_meas: np.ndarray,
        dt: float
    ) -> INSState:
        """
        EKF prediction step (strapdown mechanization + covariance propagation).
        
        Args:
            state: Current INSState.
            gyro_meas: Measured angular velocity (rad/s), shape (3,).
            accel_meas: Measured specific force (m/s²), shape (3,).
            dt: Time step (s).
        
        Returns:
            Predicted INSState.
        """
        # Correct measurements for estimated biases
        gyro_corrected = gyro_meas - state.b_g
        accel_corrected = accel_meas - state.b_a
        
        # Strapdown propagation (nominal state)
        q_new, v_new, p_new = strapdown_update(
            state.q, state.v, state.p,
            gyro_corrected, accel_corrected,
            dt, g=self.g, frame=self.frame, lat_rad=self.lat_rad
        )
        
        # Normalize quaternion
        q_new = quat_normalize(q_new)
        
        # Bias propagation (constant bias model)
        b_g_new = state.b_g  # Biases are constant (random walk model)
        b_a_new = state.b_a
        
        # Covariance propagation (simplified)
        # For a full implementation, would need F (state transition Jacobian)
        # Here we use a simplified model with additive process noise
        P_new = state.P + self.compute_process_noise(dt)
        
        return INSState(q=q_new, v=v_new, p=p_new, b_g=b_g_new, b_a=b_a_new, P=P_new)
    
    def compute_process_noise(self, dt: float) -> np.ndarray:
        """
        Compute process noise covariance Q for discrete time step.
        
        This is a simplified model. A full implementation would compute
        Q from continuous-time noise and integrate over dt.
        
        State ordering: [p, v, q, b_g, b_a]
        
        Args:
            dt: Time step (s).
        
        Returns:
            Process noise covariance Q, shape (16, 16).
        """
        Q = np.zeros((16, 16))
        
        # Position noise (indices 0:3, integrated from velocity noise)
        v_noise = (self.imu_params.accel_vrw_mps_sqrt_s * np.sqrt(dt))**2
        p_noise = v_noise * dt**2
        Q[0:3, 0:3] = np.eye(3) * p_noise
        
        # Velocity noise (indices 3:6, from accel VRW)
        Q[3:6, 3:6] = np.eye(3) * v_noise * dt
        
        # Attitude noise (indices 6:10, from gyro ARW)
        q_noise = (self.imu_params.gyro_arw_rad_sqrt_s * np.sqrt(dt))**2
        Q[6:10, 6:10] = np.eye(4) * q_noise * 0.1  # Scale down for quaternion
        
        # Gyro bias random walk (indices 10:13, very slow drift)
        bg_noise = (self.imu_params.gyro_bias_rad_s * 0.01 * np.sqrt(dt))**2
        Q[10:13, 10:13] = np.eye(3) * bg_noise
        
        # Accel bias random walk (indices 13:16, very slow drift)
        ba_noise = (self.imu_params.accel_bias_mps2 * 0.01 * np.sqrt(dt))**2
        Q[13:16, 13:16] = np.eye(3) * ba_noise
        
        return Q
    
    def update_zupt(self, state: INSState) -> INSState:
        """
        EKF update step with ZUPT measurement (Eqs. 6.40-6.43, 6.45).
        
        Applies zero-velocity pseudo-measurement:
            z_k = [0, 0, 0]^T (measured velocity)
            h(x) = v (predicted velocity from state)
            H = [0_3x3, I_3, 0_3x4, 0_3x3, 0_3x3] (Jacobian, Eq. 6.45)
        
        State ordering (Eq. 6.16): x = [p (3), v (3), q (4), b_g (3), b_a (3)]
        
        Args:
            state: Predicted INSState.
        
        Returns:
            Updated INSState.
        """
        # ZUPT measurement model (Eq. 6.45)
        z_k = np.zeros(3)  # Zero velocity measurement
        h_x = state.v  # Predicted velocity
        
        # Measurement Jacobian H (extracts velocity from state)
        # State ordering: [p (3), v (3), q (4), b_g (3), b_a (3)]
        # H = [0_3x3, I_3, 0_3x4, 0_3x3, 0_3x3]
        H = np.zeros((3, 16))
        H[:, 3:6] = np.eye(3)  # ∂h/∂v = I (velocity at indices 3:6)
        
        # Measurement noise covariance R
        R = (self.sigma_zupt**2) * np.eye(3)
        
        # Innovation covariance S (Eq. 6.40)
        S = H @ state.P @ H.T + R
        
        # Kalman gain K (Eq. 6.41)
        K = state.P @ H.T @ np.linalg.inv(S)
        
        # Innovation (measurement residual)
        innovation = z_k - h_x
        
        # State update (Eq. 6.42)
        x_vec = state.to_vector()
        x_vec_new = x_vec + K @ innovation
        
        # Covariance update (Eq. 6.43)
        # Joseph form for numerical stability: P = (I - KH)P(I - KH)^T + KRK^T
        I_KH = np.eye(16) - K @ H
        P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry
        P_new = 0.5 * (P_new + P_new.T)
        
        # Convert back to INSState
        state_new = INSState.from_vector(x_vec_new, P_new)
        
        # Normalize quaternion after update
        state_new.q = quat_normalize(state_new.q)
        
        return state_new

