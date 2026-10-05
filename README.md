# Tracept
JIT-compile-time utilities for cleaner JAX code, with extra utilities for dynamic systems

# How it works
Tracept uses Python metaclasses to process your class into one that can be decomposed automatically into static (any pytree) + mutable parts (typical JAX arrays). The mutable parts can be modified similarly to vanilla Python such as s.x += 1, which would normally be something like s = tree_set(s, 'x', s.x+1). This functionality is only available if using tracept.jit or manually .frozen() before calling a jit function then .live() after entering.

# Copying and replacing children

Children such as `parent.child` are views into their parent's mutable storage. Use
`fresh_like` to make an independent child with its original mutable defaults:

```python
fresh_child = tracept.fresh_like(parent.child)
```

Only that child's subtree, metadata, and mutable arrays are copied. The default
batch shape is `()`; pass `batch_shape=...` to create batched state. To preserve
current mutable values instead, use `copy.copy(parent.child)` or
`copy.deepcopy(parent.child)`. Copies of indexed views preserve the selected values.

`parent.child.node` or `parent.node.child` gives the underlying dataclass, including static attributes and
internal mutable IDs. Saving it in a regular Python dictionary is fine for
reading static attributes, but it is a reference, not an independent copy, and
it does not carry mutable values or metadata. Passing a raw `.node` as a child
to a constructor or assigning it to a Live parent raises an error; pass the Live
child or a `Child.new()` instance instead. Reusing a Live child in constructors
is safe: each parent adopts an independent subtree using its declared defaults.

Replace children outside JAX tracing with:

```python
parent.child = Child.new()
# Or adopt an independent copy of an existing child's current state:
parent.child = other.child
```

Replacement rebuilds shapes, labels, defaults, and mutable IDs, removes the old
child's storage, and preserves other state and sibling views. Replacement arrays
must broadcast to the parent's batch shape. An indexed parent cannot replace a
child, and replacement inside `tracept.jit`, `jax.jit`, or other JAX transforms
raises an error. `tracept.jit` recompiles after a replacement. Views of the
replaced subtree become invalid; obtain the child from its parent again. Copies
and frozen snapshots made beforehand remain independent.

Run the core tests with `python -m unittest discover -s tests -v`.
