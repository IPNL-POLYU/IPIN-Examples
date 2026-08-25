"""An AOA bearing has a direction, and it is easy to draw it backwards.

`aoa_azimuth(anchor, agent)` returns the azimuth measured **from the agent
toward the anchor** -- its own docstring says so, and the implementation uses
`dE = anchor_E - agent_E`. So the unit vector `(sin psi, cos psi)` points from
the agent to the anchor, and drawing it outward *from the anchor* points away
from the agent.

`ch4_aoa_geometry.png` did exactly that. All four bearing rays left the plot on
the far side of their anchor, and none passed through the position the figure
exists to show them intersecting at. The figure had been committed that way,
and no test looked: the ones here check that files were written.

This pins the convention at the library level rather than in the plotting code,
because the sign is what a reader gets wrong -- once when drawing the ray, and
again if they build their own solver from it.

Author: Li-Ta Hsu
References: Chapter 4, Eq. (4.64)
"""

import unittest

import numpy as np

from core.rf.measurement_models import aoa_azimuth

ANCHORS = np.array([[0, 0], [12, 0], [12, 12], [0, 12]], dtype=float)
AGENT = np.array([5.0, 7.0])


class TestAoaBearingDirection(unittest.TestCase):

    def test_psi_is_the_bearing_from_the_agent_to_the_anchor(self):
        """(sin psi, cos psi) walks from the agent to the anchor, not back."""
        for anchor in ANCHORS:
            psi = aoa_azimuth(anchor, AGENT)
            direction = np.array([np.sin(psi), np.cos(psi)])
            span = anchor - AGENT

            with self.subTest(anchor=tuple(anchor)):
                np.testing.assert_allclose(
                    direction,
                    span / np.linalg.norm(span),
                    atol=1e-9,
                    err_msg="psi does not point from the agent toward the anchor",
                )

    def test_the_ray_drawn_from_an_anchor_must_be_negated(self):
        """The step a geometry figure has to take, asserted.

        Walking from the anchor along -(sin psi, cos psi) reaches the agent;
        walking along +(sin psi, cos psi) doubles the distance away from it.
        """
        for anchor in ANCHORS:
            psi = aoa_azimuth(anchor, AGENT)
            toward = np.array([-np.sin(psi), -np.cos(psi)])
            distance = float(np.linalg.norm(anchor - AGENT))

            with self.subTest(anchor=tuple(anchor)):
                np.testing.assert_allclose(
                    anchor + distance * toward,
                    AGENT,
                    atol=1e-9,
                    err_msg="the anchor-to-agent ray does not reach the agent",
                )
                wrong = anchor + distance * (-toward)
                self.assertGreater(
                    float(np.linalg.norm(wrong - AGENT)),
                    distance,
                    "the un-negated ray should move away from the agent",
                )


if __name__ == "__main__":
    unittest.main()
