import unittest

import jax.numpy as jnp
import numpy as np

from tracept import Tracept, Mutable
from tracept.interp import LabelWrapper, interp_class


class InterpChild(metaclass=Tracept):
    x: Mutable(shape=(2,)) = None


class InterpParent(metaclass=Tracept):
    child: InterpChild = None
    values: jnp.ndarray = None
    name: str = 'sample'


class TestInterp(unittest.TestCase):
    def test_label_format(self):
        value = LabelWrapper(jnp.array([1.25, 2.5]), {'a': 0, 'b': 1})
        self.assertEqual(repr(value), '(a=1.25, b=2.5)')
        self.assertEqual(format(value, '.2f'), '(a=1.25, b=2.50)')
        with self.assertRaises(ValueError):
            format(value, 'invalid')

    def test_interpolated_repr_and_field_selection(self):
        child = InterpChild.new()
        parent = InterpParent(child=child, values=jnp.array([10.0, 20.0]))
        parent.child.x = jnp.array([2.0, 6.0])
        value = interp_class(0.25, jnp.array([0.0, 1.0]), parent)
        np.testing.assert_allclose(value.child.x, 3.0)
        self.assertEqual(repr(value), 'InterpParent( child=InterpChild( x=3.,  ), values=12.5, name=sample,  )')
        self.assertEqual(format(value, 'tm'), 'InterpParent( child=InterpChild( x=3.,  ),  )')
        self.assertEqual(format(value, 'l'), 'InterpParent( values=12.5, name=sample,  )')
        self.assertEqual(format(value, ''), 'InterpParent(  )')
        with self.assertRaisesRegex(ValueError, '"q" is not a recognized format specifier'):
            format(value, 'q')


if __name__ == '__main__':
    unittest.main()
