import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from tracept import Tracept, Mutable, fresh_like, jit


class Child(metaclass=Tracept, static_attrnames=['scale']):
    x: Mutable(default=2, labels=['states']) = None
    scale: int = 3


class VectorChild(metaclass=Tracept):
    x: Mutable(default=4, shape=(2,), labels=['vectors']) = None
    y: Mutable(default=5, labels=['states']) = None


class StaticChild(metaclass=Tracept):
    value: int = 7


class Parent(metaclass=Tracept):
    x: Mutable(default=1, labels=['states']) = None
    child: Child = None
    sibling: Child = None

    @classmethod
    def new(cls, **kwargs):
        return cls(child=Child.new(), sibling=Child.new(), **kwargs)


class Grandparent(metaclass=Tracept):
    x: Mutable(default=9, labels=['outer']) = None
    parent: Parent = None


class EmptyParent(metaclass=Tracept):
    child: Child = None


class Collection(metaclass=Tracept):
    children: object = None


class TypedChild(metaclass=Tracept):
    count: Mutable(default=2, dtype=int, labels=['counts']) = None
    flags: Mutable(default=True, shape=(2,), dtype=bool, labels=['flags']) = None
    empty: Mutable(shape=(2,), dtype=int) = None
    value: Mutable(default=1) = None


class TestCore(unittest.TestCase):
    def assertArrayEqual(self, actual, expected):
        np.testing.assert_array_equal(actual, expected)

    def test_mutable_dtypes_and_batched_assignment(self):
        child = TypedChild.new(batch_shape=3)
        self.assertEqual(child.count.dtype, jnp.asarray(0).dtype)
        self.assertEqual(child.flags.dtype, jnp.bool_)
        self.assertEqual(child.value.dtype, jnp.asarray(0.).dtype)
        self.assertArrayEqual(child.count, [2, 2, 2])
        self.assertArrayEqual(child.flags, np.ones((3, 2), dtype=bool))
        self.assertArrayEqual(child.empty, np.zeros((3, 2), dtype=int))
        child[1].count = 7
        child[0]['flags', 0] = [False, True]
        self.assertArrayEqual(child.count, [2, 7, 2])
        self.assertArrayEqual(child.flags[0], [False, True])

    def test_mutable_dtypes_survive_jit(self):
        @jit
        def update(child):
            child.count += 1
            child.flags = ~child.flags
            return child.count, child.flags

        child = TypedChild.new()
        for expected in [3, 4]:
            count, flags = update(child)
            self.assertEqual(count.dtype, jnp.asarray(0).dtype)
            self.assertEqual(flags.dtype, jnp.bool_)
            self.assertArrayEqual(count, expected)
            self.assertArrayEqual(flags, [expected == 4]*2)

    def test_mutable_dtype_metadata_survives_child_lifecycle(self):
        child = TypedChild.new()
        parent = EmptyParent.new(child=child)
        collection = Collection.new(children=[child])
        parent.child.count = 9
        parent.child.flags = [False, True]
        parent.child = copy.copy(parent.child)
        for restored in [parent.child, parent.child.frozen().live(),
                         copy.copy(parent.child), collection.children[0]]:
            self.assertEqual(restored.count.dtype, jnp.asarray(0).dtype)
            self.assertEqual(restored.flags.dtype, jnp.bool_)
            fresh = fresh_like(restored, batch_shape=2)
            self.assertEqual(fresh.count.dtype, restored.count.dtype)
            self.assertEqual(fresh.flags.dtype, restored.flags.dtype)
            self.assertArrayEqual(fresh.count, [2, 2])
            self.assertArrayEqual(fresh.flags, np.ones((2, 2), dtype=bool))

    def test_plain_attributes_can_be_modified_outside_jit(self):
        child = StaticChild.new()
        frozen = child.frozen()
        copied = copy.copy(child)
        child.value += 1
        self.assertEqual(child.value, 8)
        self.assertEqual(child.node.value, 8)
        self.assertEqual(frozen.live().value, 7)
        self.assertEqual(copied.value, 7)
        self.assertEqual(len(child.box.muts), 0)

        parent = EmptyParent.new(child=child, batch_shape=3)
        view = parent.child
        parent[0].child.value = 9
        self.assertEqual(view.value, 9)
        self.assertEqual(parent[1].child.value, 9)
        self.assertEqual(child.value, 8)

    def test_plain_attribute_assignment_reuses_jit(self):
        traces = []

        @jit
        def read(child):
            traces.append(None)
            return child.value

        child = StaticChild.new()
        self.assertArrayEqual(read(child), 7)
        self.assertArrayEqual(read(child), 7)
        child.value = 10
        self.assertArrayEqual(read(child), 10)
        self.assertArrayEqual(read(child), 10)
        other = StaticChild.new(value=11)
        self.assertArrayEqual(read(other), 11)
        self.assertEqual(len(traces), 1)

    def test_nested_static_attribute_assignment_retraces_jit(self):
        traces = []

        @jit
        def update(parent):
            traces.append(parent.child.scale)
            parent.child.x += parent.child.scale
            return parent.child.x

        parent = Parent.new()
        self.assertArrayEqual(update(parent), 5)
        parent.child.scale = 10
        self.assertArrayEqual(update(parent), 15)
        self.assertArrayEqual(update(parent), 25)
        self.assertEqual(traces, [3, 10])
        parent.child.scale = 3
        self.assertArrayEqual(update(parent), 28)
        self.assertEqual(traces, [3, 10])

    def test_nested_plain_attribute_assignment_reuses_jit(self):
        traces = []

        @jit
        def read(parent):
            traces.append(None)
            return parent.child.value, parent.child

        parent = EmptyParent.new(child=StaticChild.new())
        value, child = read(parent)
        self.assertArrayEqual(value, 7)
        self.assertArrayEqual(child.value, 7)
        parent.child.value = 10
        value, child = read(parent)
        self.assertArrayEqual(value, 10)
        self.assertArrayEqual(child.value, 10)
        self.assertEqual(len(traces), 1)

    def test_plain_attribute_shape_and_dtype_changes_retrace_jit(self):
        traces = []

        @jit
        def read(child):
            traces.append(None)
            return child.value * 2

        child = StaticChild.new(value=jnp.ones(2, dtype=jnp.float32))
        self.assertArrayEqual(read(child), [2, 2])
        child.value = jnp.full((2,), 3, dtype=jnp.float32)
        self.assertArrayEqual(read(child), [6, 6])
        self.assertEqual(len(traces), 1)
        child.value = jnp.ones(3, dtype=jnp.float32)
        self.assertArrayEqual(read(child), [2, 2, 2])
        self.assertEqual(len(traces), 2)
        child.value = jnp.ones(3, dtype=jnp.int32)
        self.assertArrayEqual(read(child), [2, 2, 2])
        self.assertEqual(len(traces), 3)

    def test_child_structure_changes_retrace_and_reuse_jit(self):
        traces = []

        @jit
        def read(parent):
            traces.append(type(parent.child.node))
            return parent.child.x

        parent = EmptyParent.new(child=Child.new())
        self.assertArrayEqual(read(parent), 2)
        parent.child = Child.new()
        self.assertArrayEqual(read(parent), 2)
        self.assertEqual(traces, [Child])
        parent.child = VectorChild.new()
        self.assertArrayEqual(read(parent), [4, 4])
        self.assertEqual(traces, [Child, VectorChild])
        parent.child = Child.new()
        self.assertArrayEqual(read(parent), 2)
        self.assertEqual(traces, [Child, VectorChild])

    def test_plain_attribute_changes_reuse_gradient_modes(self):
        for mode in ('grad', 'vng'):
            with self.subTest(mode=mode):
                traces = []

                @jit(mode=mode)
                def evaluate(child, x):
                    traces.append(None)
                    return child.value * x**2

                child = StaticChild.new()
                result = evaluate(child, 2.)
                if mode == 'vng':
                    self.assertArrayEqual(result[0], 28)
                    result = result[1]
                self.assertArrayEqual(result, 28)
                child.value = 10
                result = evaluate(child, 2.)
                if mode == 'vng':
                    self.assertArrayEqual(result[0], 40)
                    result = result[1]
                self.assertArrayEqual(result, 40)
                self.assertEqual(len(traces), 1)

    def test_plain_attributes_work_with_multiple_live_keyword_arguments(self):
        traces = []

        @jit
        def update(parent, child, offset):
            traces.append(None)
            parent.x += child.value + offset
            return parent.x

        parent, child = Parent.new(), StaticChild.new()
        self.assertArrayEqual(update(parent, child=child, offset=2), 10)
        child.value = 10
        self.assertArrayEqual(update(parent, child=child, offset=3), 23)
        self.assertEqual(len(traces), 1)

    def test_plain_attribute_assignment_is_rejected_during_tracing(self):
        child = StaticChild.new()

        @jit
        def update(child):
            child.value = 8

        @jax.jit
        def native(frozen):
            live = frozen.live()
            live.value = 8
            return live.frozen()

        def captured(x):
            child.value = 8
            return x

        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            update(child)
        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            native(child.frozen())
        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            jax.grad(captured)(1.)
        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            jax.vmap(captured)(jnp.ones(2))
        self.assertEqual(child.value, 7)

    def test_plain_assignment_cannot_discard_mutable_child_storage(self):
        parent = Parent.new()
        with self.assertRaisesRegex(ValueError, 'Live child'):
            parent.child = None
        with self.assertRaisesRegex(ValueError, 'Live child'):
            parent.child.scale = parent.child.node.x
        self.assertArrayEqual(parent.child.x, 2)
        self.assertEqual(parent.child.scale, 3)

    def test_fresh_child_has_local_metadata_and_defaults(self):
        parent = Parent.new()
        parent.child.x = 20
        child = fresh_like(parent.child)
        self.assertIs(type(child.node), Child)
        self.assertEqual(child.node.x.i, 0)
        self.assertEqual(child.box.meta.mut_shapes, [()])
        self.assertEqual(child.box.meta.defaults, {0: 2})
        self.assertEqual(len(child['states']), 1)
        self.assertArrayEqual(child.x, 2)
        child.x = 30
        self.assertArrayEqual(parent.child.x, 20)
        child.node.scale = 8
        self.assertEqual(parent.child.scale, 3)

    def test_fresh_grandchild_excludes_ancestors_and_siblings(self):
        root = Grandparent.new(parent=Parent.new())
        child = fresh_like(root.parent.child, batch_shape=3)
        self.assertEqual(len(child.box.muts), 1)
        self.assertArrayEqual(child.x, [2, 2, 2])
        self.assertEqual(child.box.batch_shape, (3,))
        self.assertNotIn('outer', child.box.meta.labeled_mut_ids)

    def test_fresh_root_is_independent(self):
        original = Parent.new()
        fresh = fresh_like(original)
        fresh.child = VectorChild.new()
        self.assertIs(type(original.child.node), Child)
        self.assertEqual(len(original.box.muts), 3)
        self.assertArrayEqual(original.sibling.x, 2)

    def test_copy_child_preserves_state_and_index(self):
        parent = Parent.new(batch_shape=3)
        parent.child.x = jnp.array([10., 20., 30.])
        for copier in (copy.copy, copy.deepcopy):
            child = copier(parent.child)
            self.assertEqual(len(child.box.muts), 1)
            self.assertArrayEqual(child.x, [10, 20, 30])
            child.x = 99
            self.assertArrayEqual(parent.child.x, [10, 20, 30])
            indexed = copier(parent[1].child)
            self.assertArrayEqual(indexed.x, 20)
            self.assertEqual(indexed.box.batch_shape, ())

    def test_reusing_live_children_does_not_change_source_ids(self):
        child = Child.new()
        child.x = 40
        first = Parent(child=child, sibling=child)
        second = Parent(child=child, sibling=child)
        self.assertEqual(child.node.x.i, 0)
        self.assertArrayEqual(child.x, 40)
        # Construction retains declared defaults; use copy for a state snapshot.
        self.assertArrayEqual(first.child.x, 2)
        first.child.x = 7
        self.assertArrayEqual(first.sibling.x, 2)
        self.assertArrayEqual(second.child.x, 2)
        self.assertEqual([mid.i for mid in first.box.meta.labeled_mut_ids['states']], [0, 1, 2])

    def test_adopt_existing_child_view_only_imports_its_metadata(self):
        source = Grandparent.new(parent=Parent.new())
        parent = Parent(child=source.parent.child, sibling=Child.new())
        self.assertEqual(len(parent.box.muts), 3)
        self.assertNotIn('outer', parent.box.meta.labeled_mut_ids)
        self.assertEqual(source.parent.child.node.x.i, 2)

    def test_node_baking_is_rejected(self):
        # Biscuit detection (baking twice should raise an error)
        with self.assertRaisesRegex(ValueError, r'not \.node'):
            Parent(child=Child.new().node, sibling=Child.new())
        with self.assertRaisesRegex(ValueError, r'not \.node'):
            Collection(children=[Child.new().node])

    def test_replacement_rebuilds_metadata_and_preserves_other_views(self):
        parent = Parent.new()
        sibling = parent.sibling
        old_child = parent.child
        parent.x = 10
        sibling.x = 30
        replacement = VectorChild.new()
        replacement.y = 50
        parent.child = replacement
        self.assertEqual(parent.box.meta.mut_shapes, [(), (), (2,), ()])
        self.assertEqual(parent.box.meta.defaults, {0: 1, 1: 2, 2: 4, 3: 5})
        self.assertArrayEqual(parent.x, 10)
        self.assertArrayEqual(sibling.x, 30)
        self.assertArrayEqual(parent.child.x, [4, 4])
        self.assertArrayEqual(parent.child.y, 50)
        self.assertEqual([mid.i for mid in parent.box.meta.labeled_mut_ids['states']], [0, 1, 3])
        self.assertEqual(parent.box.meta.labeled_mut_ids['vectors'][0].i, 2)
        sibling.x = 31
        self.assertArrayEqual(parent.sibling.x, 31)
        parent.child.y = 51
        self.assertArrayEqual(replacement.y, 50)
        with self.assertRaisesRegex(ValueError, 'replaced'):
            _ = old_child.x
        with self.assertRaisesRegex(ValueError, 'replaced'):
            old_child.x = 1
        with self.assertRaisesRegex(ValueError, 'replaced'):
            fresh_like(old_child)

    def test_repeated_replacement_removes_unused_state_and_labels(self):
        parent = Parent.new()
        for _ in range(3):
            parent.child = VectorChild.new()
            self.assertEqual(len(parent.box.muts), 4)
            parent.child = Child.new()
            self.assertEqual(len(parent.box.muts), 3)
            self.assertNotIn('vectors', parent.box.meta.labeled_mut_ids)
        parent.child = StaticChild.new()
        self.assertEqual(len(parent.box.muts), 2)
        self.assertEqual(parent.child.value, 7)

    def test_nested_replacement_preserves_root_and_sibling(self):
        root = Grandparent.new(parent=Parent.new())
        sibling = root.parent.sibling
        root.x = 100
        sibling.x = 30
        root.parent.child = VectorChild.new()
        self.assertEqual(len(root.box.muts), 5)
        self.assertArrayEqual(root.x, 100)
        self.assertArrayEqual(sibling.x, 30)
        self.assertArrayEqual(root.parent.child.x, [4, 4])
        root.parent.child = Child.new()
        self.assertEqual(len(root.box.muts), 4)
        self.assertArrayEqual(sibling.x, 30)

    def test_replacement_from_own_sibling_or_child_is_independent(self):
        parent = Parent.new()
        parent.sibling.x = 17
        parent.child = parent.sibling
        self.assertArrayEqual(parent.child.x, 17)
        parent.child.x = 18
        self.assertArrayEqual(parent.sibling.x, 17)
        parent.child = parent.child
        self.assertArrayEqual(parent.child.x, 18)

    def test_node_snapshot_keeps_static_values_after_replacement(self):
        parent = Parent.new()
        old_statics = {'child': parent.node.child}
        parent.child = Child.new(scale=8)
        self.assertEqual(old_statics['child'].scale, 3)
        self.assertEqual(parent.child.scale, 8)
        with self.assertRaisesRegex(ValueError, r'not \.node'):
            parent.child = old_statics['child']

    def test_replacement_broadcasts_to_parent_batch(self):
        parent = Parent.new(batch_shape=(3,))
        parent.child = VectorChild.new()
        self.assertArrayEqual(parent.child.x, np.full((3, 2), 4))
        self.assertArrayEqual(parent.child.y, [5, 5, 5])
        parent.child = Child.new(batch_shape=3)
        self.assertArrayEqual(parent.child.x, [2, 2, 2])

    def test_invalid_batch_replacement_is_atomic(self):
        parent = Parent.new(batch_shape=3)
        node, meta, muts = parent.child.node, parent.box.meta, parent.box.muts
        with self.assertRaises(ValueError):
            parent.child = Child.new(batch_shape=2)
        self.assertIs(parent.child.node, node)
        self.assertIs(parent.box.meta, meta)
        self.assertIs(parent.box.muts, muts)
        self.assertArrayEqual(parent.child.x, [2, 2, 2])

    def test_indexed_parent_cannot_replace_child(self):
        parent = Parent.new(batch_shape=3)
        with self.assertRaisesRegex(ValueError, 'unindexed'):
            parent[0].child = Child.new()
        self.assertArrayEqual(parent.child.x, [2, 2, 2])

    def test_replacement_of_empty_child(self):
        parent = Parent(child=None, sibling=Child.new(), batch_shape=3)
        parent.child = Child.new()
        self.assertArrayEqual(parent.child.x, [2, 2, 2])

    def test_static_parent_can_gain_mutables(self):
        parent = EmptyParent.new(child=StaticChild.new(), batch_shape=3)
        self.assertEqual(parent.box.batch_shape, (3,))
        self.assertEqual(len(parent.box.muts), 0)
        parent.child = Child.new()
        self.assertArrayEqual(parent.child.x, [2, 2, 2])

    def test_frozen_snapshot_can_be_reused(self):
        frozen = Parent.new().frozen()
        first = frozen.live()
        first.x = 10
        first.child = VectorChild.new()
        second = frozen.live()
        self.assertArrayEqual(second.x, 1)
        self.assertArrayEqual(second.child.x, 2)
        self.assertIs(type(second.child.node), Child)

    def test_stale_views_cannot_be_passed_to_jit_or_formatted(self):
        parent = Parent.new()
        old = parent.child
        parent.child = VectorChild.new()

        @jit
        def read(child):
            return child.x

        with self.assertRaisesRegex(ValueError, 'replaced'):
            read(old)
        with self.assertRaisesRegex(ValueError, 'replaced'):
            repr(old)

    def test_indexed_label_gaps_survive_copy_and_replacement(self):
        class Labeled(metaclass=Tracept):
            x: Mutable(default=6, labels=[('indexed', 1)]) = None

        parent = Parent(child=Labeled.new(), sibling=Child.new())
        child = fresh_like(parent.child)
        self.assertIsNone(child.box.meta.labeled_mut_ids['indexed'][0])
        self.assertArrayEqual(child['indexed', 1], 6)
        parent.child = Labeled.new()
        self.assertIsNone(parent.box.meta.labeled_mut_ids['indexed'][0])
        self.assertArrayEqual(parent['indexed', 1], 6)
        parent.child = Child.new()
        self.assertNotIn('indexed', parent.box.meta.labeled_mut_ids)

    def test_replacement_under_other_jax_transforms_is_rejected(self):
        parent = Parent.new()

        def replace(x):
            parent.child = Child.new()
            return x

        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            jax.grad(replace)(1.)
        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            jax.vmap(replace)(jnp.ones(2))

    def test_containers_support_child_copy_and_replacement(self):
        source = Child.new()
        for children in ([source], (source,), {'a': source}):
            parent = Collection(children=children)
            child = parent.children['a' if type(children) is dict else 0]
            self.assertArrayEqual(child.x, 2)
            self.assertEqual(len(fresh_like(child).box.muts), 1)
        root = Collection(children=[Parent.new(), Parent.new()])
        sibling = root.children[1].child
        root.children[0].child = VectorChild.new()
        sibling.x = 12
        self.assertArrayEqual(root.children[1].child.x, 12)
        self.assertArrayEqual(root.children[0].child.x, [4, 4])

    def test_frozen_child_is_local_and_snapshot_survives_replacement(self):
        parent = Parent.new()
        parent.child.x = 20
        frozen = parent.frozen()
        child = parent.child.frozen()
        self.assertEqual(len(child.muts), 1)
        self.assertEqual(child.tin.x.i, 0)
        parent.child = VectorChild.new()
        self.assertArrayEqual(frozen.live().child.x, 20)
        self.assertArrayEqual(child.live().x, 20)

    def test_replacement_retraces_tracept_jit(self):
        traces = []

        @jit
        def update(parent):
            traces.append(parent.child.scale)
            parent.child.x += parent.child.scale
            return parent.child.x

        parent = Parent.new()
        self.assertArrayEqual(update(parent), 5)
        self.assertArrayEqual(update(parent), 8)
        parent.child = Child.new(scale=10)
        self.assertArrayEqual(update(parent), 12)
        self.assertArrayEqual(update(parent), 22)
        self.assertEqual(traces, [3, 10])
        other = Parent.new()
        self.assertArrayEqual(update(other), 5)

    def test_jit_multiple_live_arguments(self):
        @jit
        def update(first, second):
            first.x += 1
            second.child.x += 2
            return second.child

        first, second = Child.new(), Parent.new()
        child = update(first, second)
        self.assertArrayEqual(first.x, 3)
        self.assertArrayEqual(second.child.x, 4)
        self.assertEqual(len(child.box.muts), 1)
        self.assertArrayEqual(child.x, 4)

    def test_replacement_is_rejected_during_tracept_and_native_jit(self):
        parent = Parent.new()

        @jit
        def replace(parent):
            parent.child = Child.new()

        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            replace(parent)

        @jax.jit
        def native(frozen):
            live = frozen.live()
            live.child = Child.new()
            return live.frozen()

        with self.assertRaisesRegex(ValueError, 'outside JAX tracing'):
            native(parent.frozen())
        self.assertArrayEqual(parent.child.x, 2)

    def test_frozen_child_works_in_native_jit(self):
        @jax.jit
        def update(frozen):
            child = frozen.live()
            child.x += 1
            return child.frozen()

        parent = Parent.new()
        child = update(parent.child.frozen()).live()
        self.assertArrayEqual(child.x, 3)
        self.assertArrayEqual(parent.child.x, 2)


if __name__ == '__main__':
    unittest.main()
