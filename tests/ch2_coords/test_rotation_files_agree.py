"""The three rotation files in the Ch2 dataset must describe one rotation.

`euler_angles.txt`, `quaternions.txt` and `rotation_matrices.txt` are three
representations of the same 20 rotations, and the point of shipping all three
is that a reader can convert between them and get the same answer.

They did not. The committed quaternions were generated as
`euler_to_quat(pitch, roll, yaw)` -- roll and pitch swapped -- and the
committed matrices as `euler_to_rotation_matrix(pitch, roll, yaw).T`, the same
swap plus a transpose. So the two derived files disagreed with the Euler angles
and with each other: converting the quaternion to a matrix and comparing gave
rotations 27.8 to 169.3 degrees apart, median 120.8.

Nothing caught it for the reason this repo keeps rediscovering. The README's
"Compare Rotation Representations" experiment recomputes both representations
from `euler_angles.txt` and compares those, which passes whatever the shipped
files contain. The example printed the dataset quaternion beside the computed
one under the heading that invites the comparison, and asserted nothing. No
test read the two derived files at all.

Tolerance is set by the files, not the maths: both are stored at `fmt="%.6f"`,
so element-wise round-off is about 2e-6 and any check tighter than that fails
on quantisation rather than on content.

Author: Li-Ta Hsu
References: Chapter 2, rotation representations
"""

import unittest
from pathlib import Path

import numpy as np

from core.coords import (
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_rotation_matrix,
)

DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "sim" / "ch2_coords_san_francisco"
)

#: Files store 6 decimal places, so ~2e-6 is the floor for any comparison.
STORAGE_TOL = 1e-5


def _load():
    euler = np.loadtxt(DATA_DIR / "euler_angles.txt")
    quats = np.loadtxt(DATA_DIR / "quaternions.txt")
    mats = np.loadtxt(DATA_DIR / "rotation_matrices.txt").reshape(-1, 3, 3)
    return euler, quats, mats


class TestCh2RotationFilesAgree(unittest.TestCase):
    """Cross-file checks: the shipped bytes, not a recomputation of them."""

    def test_the_three_files_describe_the_same_rotations(self):
        euler, quats, mats = _load()
        self.assertEqual(len(euler), len(quats))
        self.assertEqual(len(euler), len(mats))

        for i, (roll, pitch, yaw) in enumerate(euler):
            with self.subTest(point=i):
                np.testing.assert_allclose(
                    quats[i],
                    euler_to_quat(roll, pitch, yaw),
                    atol=STORAGE_TOL,
                    err_msg="quaternion does not match its Euler angles",
                )
                np.testing.assert_allclose(
                    mats[i],
                    euler_to_rotation_matrix(roll, pitch, yaw),
                    atol=STORAGE_TOL,
                    err_msg="rotation matrix does not match its Euler angles",
                )

    def test_the_quaternion_and_matrix_files_agree_with_each_other(self):
        """The check that fails loudest when an argument order slips.

        Comparing each derived file against the Euler angles would catch a
        swap in one of them; comparing them against each other catches a swap
        applied to both, which is what happened here.
        """
        _, quats, mats = _load()

        for i in range(len(quats)):
            with self.subTest(point=i):
                np.testing.assert_allclose(
                    quat_to_rotation_matrix(quats[i]),
                    mats[i],
                    atol=STORAGE_TOL,
                    err_msg="quaternion and matrix describe different rotations",
                )

    def test_the_shipped_rotations_are_valid(self):
        """Unit quaternions, and matrices that are actually rotations."""
        _, quats, mats = _load()

        np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=STORAGE_TOL)
        for i, R in enumerate(mats):
            with self.subTest(point=i):
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=STORAGE_TOL)
                self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, delta=STORAGE_TOL)


if __name__ == "__main__":
    unittest.main()
