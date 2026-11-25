import unittest

import numpy as np

from motion_flow.keyboard_viewer import measured_twist
from motion_flow.robot import root_quat


class RootVelocityTests(unittest.TestCase):
    def test_body_axes_during_lateral_motion(self):
        quat = root_quat(.1, -.2, np.pi / 2)
        start = np.r_[1., 2., .4, quat]
        end = np.r_[.996, 1.994, .45, -quat]
        np.testing.assert_allclose(measured_twist(start, end, .02), [-.3, .2, 0.], atol=1e-12)

    def test_quarter_circle_crossing_yaw_wrap(self):
        heading = 3 * np.pi / 4
        start = np.r_[0., 0., .4, root_quat(.4, -.2, heading)]
        # A unit-radius left turn sweeps local displacement (1, 1).
        dx = np.cos(heading) - np.sin(heading)
        dy = np.sin(heading) + np.cos(heading)
        end = np.r_[dx, dy, .42, root_quat(-.2, .1, -heading)]
        np.testing.assert_allclose(measured_twist(start, end, np.pi/2), [1., 0., 1.], atol=1e-12)
