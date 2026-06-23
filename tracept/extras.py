from functools import partial
from dataclasses import dataclass

import jax

# Simply a container for a single static variable
#     avoids the use of static_argnames, which causes JAX tracer leaks in some libraries such as flax's NNX
@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=['v'])
@dataclass
class Static:
    v: any
    
    def __int__ (self): return int (self.v)
    def __str__ (self): return str (self.v)
    def __hash__(self): return hash(self.v)
    def __call__(self, *vargs, **kwargs): return self.v(*vargs, **kwargs)

    def __eq__(self, other):
        if isinstance(other, Static):
            return self.v == other.v
        else:
            return self.v == other
