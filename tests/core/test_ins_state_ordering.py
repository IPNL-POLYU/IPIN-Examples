"""
Unit tests for INS state ordering (Eq. 6.16).

This module tests that the INSState class and all measurement models correctly
implement the state ordering from Eq. (6.16): x = [p, v, q, b_g, b_a]^T.

Critical tests:
    1. INSState.to_vector() produces correct ordering
    2. INSState.from_vector() unpacks correctly
    3. ZUPT measurement Jacobian H selects velocity at correct indices
    4. ZARU measurement Jacobian H selects gyro bias at correct indices
    5. NHC measurement Jacobian H selects velocity and quaternion at correct indices

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np
from core.sensors.ins_ekf import INSState
from core.sensors.constraints import (
    ZuptMeasurementModel,
    ZaruMeasurementModelPlaceholder,
    NhcMeasurementModel,
)


class TestINSStateOrdering(unittest.TestCase):
    """Test INSState ordering matches Eq. (6.16)."""

    def setUp(self):
        """Set up test state."""
        self.p = np.array([10.0, 20.0, 30.0])
        self.v = np.array([1.0, 2.0, 3.0])
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.b_g = np.array([0.01, -0.02, 0.03])
        self.b_a = np.array([0.1, -0.2, 0.3])
        self.P = np.eye(16)

    def test_state_dimension(self):
        """Test that state dimension is 16."""
        state = INSState(
            p=self.p, v=self.v, q=self.q, b_g=self.b_g, b_a=self.b_a, P=self.P
        )
        x_vec = state.to_vector()

        self.assertEqual(x_vec.shape[0], 16, "State vector must have 16 elements")
        self.assertEqual(
            3 + 3 + 4 + 3 + 3, 16, "State dimension breakdown must sum to 16"
        )

    def test_to_vector_ordering(self):
        """Test that to_vector() produces [p, v, q, b_g, b_a] ordering."""
        state = INSState(
            p=self.p, v=self.v, q=self.q, b_g=self.b_g, b_a=self.b_a, P=self.P
        )
        x_vec = state.to_vector()

        # Check each component is in the correct position
        np.testing.assert_array_equal(
            x_vec[0:3], self.p, err_msg="Position must be at indices 0:3"
        )
        np.testing.assert_array_equal(
            x_vec[3:6], self.v, err_msg="Velocity must be at indices 3:6"
        )
        np.testing.assert_array_equal(
            x_vec[6:10], self.q, err_msg="Quaternion must be at indices 6:10"
        )
        np.testing.assert_array_equal(
            x_vec[10:13], self.b_g, err_msg="Gyro bias must be at indices 10:13"
        )
        np.testing.assert_array_equal(
            x_vec[13:16], self.b_a, err_msg="Accel bias must be at indices 13:16"
        )

    def test_from_vector_ordering(self):
        """Test that from_vector() unpacks [p, v, q, b_g, b_a] correctly."""
        # Create known vector
        x_vec = np.concatenate([self.p, self.v, self.q, self.b_g, self.b_a])

        # Unpack
        state = INSState.from_vector(x_vec, self.P)

        # Verify each component
        np.testing.assert_array_equal(
            state.p, self.p, err_msg="Position unpacked incorrectly"
        )
        np.testing.assert_array_equal(
            state.v, self.v, err_msg="Velocity unpacked incorrectly"
        )
        np.testing.assert_array_equal(
            state.q, self.q, err_msg="Quaternion unpacked incorrectly"
        )
        np.testing.assert_array_equal(
            state.b_g, self.b_g, err_msg="Gyro bias unpacked incorrectly"
        )
        np.testing.assert_array_equal(
            state.b_a, self.b_a, err_msg="Accel bias unpacked incorrectly"
        )

    def test_round_trip_consistency(self):
        """Test that to_vector() -> from_vector() is identity."""
        state_original = INSState(
            p=self.p, v=self.v, q=self.q, b_g=self.b_g, b_a=self.b_a, P=self.P
        )

        # Round trip
        x_vec = state_original.to_vector()
        state_recovered = INSState.from_vector(x_vec, self.P)

        # Verify
        np.testing.assert_array_almost_equal(
            state_recovered.p, state_original.p, decimal=10
        )
        np.testing.assert_array_almost_equal(
            state_recovered.v, state_original.v, decimal=10
        )
        np.testing.assert_array_almost_equal(
            state_recovered.q, state_original.q, decimal=10
        )
        np.testing.assert_array_almost_equal(
            state_recovered.b_g, state_original.b_g, decimal=10
        )
        np.testing.assert_array_almost_equal(
            state_recovered.b_a, state_original.b_a, decimal=10
        )


class TestZuptMeasurementJacobian(unittest.TestCase):
    """Test ZUPT measurement Jacobian H matches Eq. (6.45)."""

    def test_zupt_jacobian_selects_velocity(self):
        """Test that ZUPT Jacobian H = [0, I, 0, 0, 0] selects velocity."""
        model = ZuptMeasurementModel(sigma_zupt=0.05)

        # Create state vector [p, v, q, b_g, b_a]
        p = np.array([10.0, 20.0, 30.0])
        v = np.array([1.0, 2.0, 3.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        b_g = np.zeros(3)
        b_a = np.zeros(3)
        x = np.concatenate([p, v, q, b_g, b_a])

        # Get Jacobian
        H = model.H(x)

        # Verify shape
        self.assertEqual(H.shape, (3, 16), "H must be 3x16")

        # Verify it selects velocity (indices 3:6)
        expected_H = np.zeros((3, 16))
        expected_H[:, 3:6] = np.eye(3)
        np.testing.assert_array_equal(
            H, expected_H, err_msg="ZUPT Jacobian must select velocity at indices 3:6"
        )

    def test_zupt_h_extracts_velocity(self):
        """Test that ZUPT h(x) extracts velocity from correct indices."""
        model = ZuptMeasurementModel(sigma_zupt=0.05)

        # Create state vector [p, v, q, b_g, b_a]
        p = np.array([10.0, 20.0, 30.0])
        v = np.array([1.0, 2.0, 3.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        b_g = np.zeros(3)
        b_a = np.zeros(3)
        x = np.concatenate([p, v, q, b_g, b_a])

        # Extract velocity
        h_x = model.h(x)

        # Verify
        np.testing.assert_array_equal(
            h_x, v, err_msg="ZUPT h(x) must extract velocity from indices 3:6"
        )


class TestZaruMeasurementJacobian(unittest.TestCase):
    """Test ZARU measurement Jacobian H matches Eq. (6.60)."""

    def test_zaru_placeholder_jacobian_selects_gyro_bias(self):
        """Test that ZARU placeholder Jacobian H = [0, 0, 0, -I, 0] selects gyro bias."""
        model = ZaruMeasurementModelPlaceholder(sigma_zaru=0.01)

        # Create state vector [p, v, q, b_g, b_a]
        p = np.array([10.0, 20.0, 30.0])
        v = np.array([1.0, 2.0, 3.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        b_g = np.array([0.01, -0.02, 0.03])
        b_a = np.zeros(3)
        x = np.concatenate([p, v, q, b_g, b_a])

        # Get Jacobian
        H = model.H(x)

        # Verify shape
        self.assertEqual(H.shape, (3, 16), "H must be 3x16")

        # Verify it selects gyro bias (indices 10:13) with negative sign
        expected_H = np.zeros((3, 16))
        expected_H[:, 10:13] = -np.eye(3)
        np.testing.assert_array_equal(
            H,
            expected_H,
            err_msg="ZARU Jacobian must select gyro bias at indices 10:13 with -I",
        )


class TestNhcMeasurementJacobian(unittest.TestCase):
    """Test NHC measurement Jacobian H matches Eq. (6.61)."""

    def test_nhc_jacobian_uses_correct_indices(self):
        """Test that NHC Jacobian uses velocity (3:6) and quaternion (6:10)."""
        model = NhcMeasurementModel(sigma_lateral=0.1, sigma_vertical=0.05)

        # Create state vector [p, v, q, b_g, b_a]
        p = np.array([10.0, 20.0, 30.0])
        v = np.array([1.0, 0.0, 0.0])  # Forward velocity only
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity (no rotation)
        b_g = np.zeros(3)
        b_a = np.zeros(3)
        x = np.concatenate([p, v, q, b_g, b_a])

        # Get Jacobian
        H = model.H(x)

        # Verify shape
        self.assertEqual(H.shape, (2, 16), "H must be 2x16 for NHC")

        # Verify that H has non-zero blocks at velocity indices (3:6)
        H_v_block = H[:, 3:6]
        self.assertTrue(
            np.any(H_v_block != 0),
            "NHC Jacobian must have non-zero velocity block at indices 3:6",
        )

        # Verify that position block (0:3) is zero
        H_p_block = H[:, 0:3]
        np.testing.assert_array_equal(
            H_p_block,
            np.zeros((2, 3)),
            err_msg="NHC Jacobian position block must be zero",
        )

    def test_nhc_h_uses_correct_indices(self):
        """Test that NHC h(x) extracts velocity and quaternion from correct indices."""
        model = NhcMeasurementModel(sigma_lateral=0.1, sigma_vertical=0.05)

        # Create state vector [p, v, q, b_g, b_a]
        # With identity quaternion, body frame = map frame
        p = np.array([10.0, 20.0, 30.0])
        v_map = np.array([1.0, 2.0, 3.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        b_g = np.zeros(3)
        b_a = np.zeros(3)
        x = np.concatenate([p, v_map, q, b_g, b_a])

        # Compute h(x) - should give lateral and vertical velocity in body frame
        h_x = model.h(x)

        # For identity quaternion, body frame = map frame
        # So lateral (y) and vertical (z) in body = map
        expected_h = np.array([v_map[1], v_map[2]])
        np.testing.assert_array_almost_equal(
            h_x,
            expected_h,
            decimal=10,
            err_msg="NHC h(x) must extract correct velocity components",
        )


class TestStateOrderingConsistency(unittest.TestCase):
    """Test consistency of state ordering across all components."""

    def test_eq616_state_ordering_documentation(self):
        """Test that Eq. (6.16) ordering is documented correctly."""
        # This test documents the expected ordering
        # If this test fails, the code and/or documentation is inconsistent

        # Expected ordering from Eq. (6.16)
        expected_ordering = {
            "position": (0, 3),
            "velocity": (3, 6),
            "quaternion": (6, 10),
            "gyro_bias": (10, 13),
            "accel_bias": (13, 16),
        }

        # Verify with actual INSState
        p = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        b_g = np.array([7.0, 8.0, 9.0])
        b_a = np.array([10.0, 11.0, 12.0])
        P = np.eye(16)

        state = INSState(p=p, v=v, q=q, b_g=b_g, b_a=b_a, P=P)
        x = state.to_vector()

        # Verify each component
        np.testing.assert_array_equal(
            x[expected_ordering["position"][0] : expected_ordering["position"][1]], p
        )
        np.testing.assert_array_equal(
            x[expected_ordering["velocity"][0] : expected_ordering["velocity"][1]], v
        )
        np.testing.assert_array_equal(
            x[expected_ordering["quaternion"][0] : expected_ordering["quaternion"][1]],
            q,
        )
        np.testing.assert_array_equal(
            x[expected_ordering["gyro_bias"][0] : expected_ordering["gyro_bias"][1]],
            b_g,
        )
        np.testing.assert_array_equal(
            x[expected_ordering["accel_bias"][0] : expected_ordering["accel_bias"][1]],
            b_a,
        )

        print("\n" + "=" * 70)
        print("Eq. (6.16) State Ordering Verification")
        print("=" * 70)
        print("State vector: x = [p, v, q, b_g, b_a]^T")
        print(f"  Position (p):      indices {expected_ordering['position']}")
        print(f"  Velocity (v):      indices {expected_ordering['velocity']}")
        print(f"  Quaternion (q):    indices {expected_ordering['quaternion']}")
        print(f"  Gyro bias (b_g):   indices {expected_ordering['gyro_bias']}")
        print(f"  Accel bias (b_a):  indices {expected_ordering['accel_bias']}")
        print(f"Total dimension: 16 elements")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main(verbosity=2)

