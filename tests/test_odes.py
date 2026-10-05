import unittest

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from tracept import Tracept, Mutable
from tracept.odes import Derivative, make_fixed_explicit_integrator, step_fe


class Dynamics(metaclass=Tracept):
    x: Mutable(default=1.0, shape=(2,), dtype=jnp.float32) = None
    dx: Derivative('x') = None

    def __call__(self):
        self.dx = -self.x


class DerivativeTest(unittest.TestCase):
    def test_derivative_inherits_state_shape_and_dtype(self):
        dynamics = Dynamics.new()
        self.assertEqual(dynamics.dx.shape, (2,))
        self.assertEqual(dynamics.dx.dtype, dynamics.x.dtype)
        self.assertEqual(Derivative('x', default=2.0).default, 2.0)

        @jax.jit
        def update(frozen):
            live = frozen.live()
            live()
            return live.frozen()

        result = update(dynamics.frozen()).live()
        np.testing.assert_array_equal(result.dx, [-1.0, -1.0])

    def test_default_dtype_follows_precision_at_construction(self):
        with enable_x64(False):
            class LatePrecisionDynamics(metaclass=Tracept):
                x: Mutable(default=1.0) = None
                dx: Derivative('x') = None
                explicit: Mutable(default=2.0, dtype=jnp.float32) = None

                def __call__(self):
                    self.dx = jax.lax.cond(self.x > 0.0,
                                           lambda: -self.x, lambda: 0.0)

            single = LatePrecisionDynamics.new()
            self.assertEqual(single.x.dtype, np.dtype('float32'))

        with enable_x64(True):
            dynamics = LatePrecisionDynamics.new()
            self.assertEqual(dynamics.x.dtype, np.dtype('float64'))
            self.assertEqual(dynamics.dx.dtype, np.dtype('float64'))
            self.assertEqual(dynamics.explicit.dtype, np.dtype('float32'))
            integrate = make_fixed_explicit_integrator(step_fe)
            _, result = integrate(dynamics, 0.1, 0.2)
            np.testing.assert_allclose(result.x, [1.0, 0.9, 0.81])
            self.assertEqual(result.dx.dtype, np.dtype('float64'))

    def test_derivatives_work_in_jitted_ode_integration(self):
        integrate = make_fixed_explicit_integrator(step_fe)
        t, result = integrate(Dynamics.new(), 0.1, 0.2)
        np.testing.assert_allclose(t, [0.0, 0.1, 0.2])
        np.testing.assert_allclose(result.x, [[1.0]*2, [0.9]*2, [0.81]*2])
        np.testing.assert_allclose(result.dx, -result.x)
        self.assertEqual(result.dx.dtype, result.x.dtype)


if __name__ == '__main__':
    unittest.main()
