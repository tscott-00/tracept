"""Core Tracept functionality"""

# Authors: Thomas A. Scott https://www.scott-aero.com/

import inspect
from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

# TODO: via meta, allow int and bool dynamics natively
# TODO: allow dynamics to be stored individually in the box? then aggregates need to be stacked... unless aggregates stay as list of I then it is natural

# TODO: make clear what limitations on calling are - can wrap inside jit when jitting then remake outside, but is it good to return wrapper and have JAX pytree it?
#       issue is can't use the wrapper we entered with since immutable jax pytree... JIT needs to only see the pieces
#       #1: jit sees tmethod, takes in jax pytree Wrapper, take appart the jaxed static list into a python list, give mutable version to actual function during compilation
#       #2: custom jit takes in vanilla twrap, unpacks, passes to jax jitted wrap func that repacks sends to actual f, unpacks and returns out of jit, then it is repacked

# TODO: test on 0 muts
# TODO: nested twrap

class Tracept(type):
    # TODO: tmethod all member funcs? they can all only be called from wrap now
    def __new__(cls, name, bases, dct, **kwargs):
        return tclass(super().__new__(cls, name, bases, dct), static_attrnames=kwargs.pop('static_attrnames', []))

    def __call__(cls, *args, batch_shape=(), **kwargs):
        # Create and initialize the underlying tclass instance
        tin = super().__call__(*args, **kwargs)

        if tin.__is_baked__:
            # If object is being created using contents of an already baked tin, it is JAX filling a pytree with tracers so we leave it alone
            return tin
        else:
            tin.__is_baked__ = True
            # Bake it, i.e. build meta describing all Mutable
            meta = Meta()
            bake_branch(tin, meta)
            
            if type(batch_shape) is int:
                batch_shape = (batch_shape,)

            return Wrapper(tin, Box(meta.new_muts(batch_shape), meta), is_root=True)

# TODO: in other file
class jit:
    # TODO: allow static args etc for jax
    def __init__(self, func=None):#, *, z_out=None):
        self.is_pre_deco = func == None
        if not self.is_pre_deco:
            self.func = func
            param_names = inspect.signature(func).parameters
            self.is_member = len(param_names) > 0 and next(iter(param_names)) == 'self'

    def __call__(self, *vargs, **kwargs):
        if self.is_pre_deco:
            if len(vargs) != 1 or len(kwargs) != 0:
                raise RuntimeError('If arguments are provided to tmethod during decoration then further args are not allowed')
            return tmethod(vargs[0])

        if self.is_member:
            raise NotImplementedError('CBL')
        else:
            def outer(*v, **k):
                args = list(v) + list(k.values())
                targ_info, muts, rarg_I, rargs = [], [], [], []
                for i, arg in enumerate(args):
                    if type(arg) is Wrapper:
                        muts += arg.box.muts
                        targ_info.append((i, len(muts), arg.node, arg.box.meta))
                    else:
                        rarg_I.append(i)
                        rargs.append(arg)
                @jax.jit
                def jinner(muts, rargs, N_args=len(args), rarg_I=rarg_I, keys=k.keys()):
                    args = [None]*N_args
                    ptr_mut = 0
                    for i, N_mut, tin, meta in targ_info:
                        args[i] = Wrapper(tin, Box(muts[ptr_mut:ptr_mut+N_mut], meta))
                        ptr_mut += N_mut
                    for i, rarg in zip(rarg_I, rargs):
                        args[i] = rargs
                    _v, _k = args[:Nv], zip(keys, args[Nv:])
                    outputs = self.func(*_v, **_k)
                    muts = []
                    for i, *_ in targ_info:
                        muts += args[i].box.muts
                    return muts, outputs
                muts, outputs = jinner(muts, rargs)

                ptr_mut = 0
                for i, N_mut, tin, meta in targ_info:
                    args[i].box.muts = muts[ptr_mut:ptr_mut+N_mut]
                    ptr_mut += N_mut

                return outputs
            return outer




# Tracept function decorator
# Static functions can be called from anywhere and take and return only z
# Member functions are only called from tracept static functions and can take and return anything
# class tmethod:
#     def __init__(self, func=None):#, *, z_out=None):
#         self.is_pre_deco = func == None
#         # self.z_out = z_out
#         # TODO: specific locked attribs (no read or write)?
#             # TODO: way to do separate containers for a set of active variables? could speed up things like pont solver a lot
#         if not self.is_pre_deco:
#             self.func = func
#             param_names = inspect.signature(func).parameters
#             self.is_member = len(param_names) > 0 and next(iter(param_names)) == 'self'

#             # print('new tmethod', func, inspect.signature(func).parameters[0], inspect.ismethod(func), inspect.isfunction(func))
#             # @functools.wraps(func) # TODO: retain signature

#     # Users should call with z, internal state managers like integrators should call with z_dyn and z
#     def __call__(self, *vargs, **kwargs):
#         if self.is_pre_deco:
#             if len(vargs) != 1 or len(kwargs) != 0:
#                 raise RuntimeError('If arguments are provided to tmethod during decoration then further args are not allowed')
#             return tmethod(vargs[0])#, z_out=self.z_out)

#         if self.is_member:
#             if not 'tracept_self' in kwargs:
#                 raise ValueError('Tracept member functions must be called from within a Tracept static function as part of wrapped z')
#             tracept_self = kwargs['tracept_self']
#             del kwargs['tracept_self']
#             return self.func(tracept_self, *vargs, **kwargs)
#         else:
#             # TODO: how does jax deal with the io? should we 
#             if 'z_tree' in kwargs:
#                 result = self.func(z=Wrapper(kwargs['z_dyn'], kwargs['z_tree'], is_root=True))
#                 # TODO: allow other return values along with z?
#                 # if self.z_out:
#                 if isinstance(result, Wrapper):
#                     result = result.z_box.z_dyn
#             elif 'z' in kwargs:
#                 # If given a Wrapper, know this is nested call and don't intervene
#                 # TODO: allow generic return if nested static? take root flag for clarity?
#                 result = self.func(z=kwargs['z'])
#             else:
#                 raise ValueError('tmethods must be called with either wrapped z kwarg or both z_tree and z_dyn kwargs')
#             return result

# TODO: one modality
def tclass(cls=None, *, static_attrnames=[]):
    """Decorator to create a Tracept class, enabling functionality mutable JIT OOP.
    All attributes of the class must be annotated in dataclass convention.
    Tracept classes should be initialized by user with MyTClass.new(), which is auto-generated if not user-specified.
    This is because JAX JIT compiled dataclasses are internally copied by calling the constructor with all fields, preventing custom initializers.

    Args:
        static_attrnames: names of attributes to make static, only these attributes are guaranteed to trigger recompilation
    """
    def _tclass(cls, static_attrnames=static_attrnames):
        if not hasattr(cls, 'new'):
            setattr(cls, 'new', classmethod(lambda cls, *vargs, **kwargs: cls(*vargs, **kwargs)))
        jit_variables = []
        # if not is_dataclass(cls):
        cls.__annotations__['__is_baked__'] = bool
        setattr(cls, '__is_baked__', False)
        static_attrnames = ['__is_baked__'] + static_attrnames
        # TODO: allow non kw-only? pickle doesn't like pos dataclass args
        cls = dataclass(cls, kw_only=True) # Turn into dataclass
        fields = get_fields(cls) # Get fields (everything that was annotated)
        # TODO: Error if not annotated
        for field in fields:
            if not field.name in static_attrnames:
                jit_variables.append(field.name)
        jax.tree_util.register_dataclass(cls, data_fields=jit_variables, meta_fields=static_attrnames)
        return cls
    
    # Handle args vs no args provided flexibility
    if cls is None:
        return _tclass
    return _tclass(cls)

# Indicates a field will be part of the dynamic shape, user provides shape of data (at a given time for a single MC sample)
class Mutable:
    def __init__(self, default=None, shape=(), labels=None):
        """
        Args:
          default: recommended default when instantiating (e.g. used in func:fill but not func:zeros),
            should be broadcastable to arg:shape, must be broadcastable to arg:shape with z batch shape prepended
        """
        if labels == None: labels = []
        
        self.default = default
        if type(shape) is int:
            shape = (shape,)
        self.shape = shape
        self.labels = labels

# Dynamic and Derivative fields in a dsp_class are automatically turned into a DynamicsMap during build_z and store indices to the dynamic map
# @partial(jax.tree_util.register_dataclass, data_fields=['i'], meta_fields=[])
@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=['i'])
@dataclass
class MutableID:
    i: int #: indexes the underlying tuple of mut

    def __lt__(self, other): return self.i < other.i
    def __hash__(self): return hash(self.i)

# TODO: only labeled_mut_ids is needed during runtime... and MutableID needs to be static to use as index the way i do
@partial(jax.tree_util.register_dataclass, data_fields=['labeled_mut_ids', 'defaults'], meta_fields=['mut_shapes'])
@dataclass
class Meta:
    # batch_shape:     tuple[int]
    mut_shapes:      list[tuple[int]]               = field(default_factory=lambda:[]) #: base shape of each mutable
    labeled_mut_ids: dict[str, list[MutableID]]     = field(default_factory=lambda:{}) #: for each label, a list of identifiers
    defaults:        dict[MutableID, jtp.ArrayLike] = field(default_factory=lambda:{}) #: index arrays into underlying array to a broadcastable default
    #run: RuntimeMeta

    def append(self, mut: Mutable, default: jtp.ArrayLike) -> MutableID:
        """
        Args:
            mut the specificier for the mutable variable
            val the default value to use, user may choose to just pass in mut.default
        Returns:
            mid identifier used to obtain the mutable's value in the future
        """
        mid = MutableID(len(self.mut_shapes))
        self.mut_shapes.append(mut.shape)
        for label in mut.labels:
            if label not in self.labeled_mut_ids:
                self.labeled_mut_ids[label] = []
            self.labeled_mut_ids[label].append(mid)

        if default is None:
            default = mut.default
        if default is not None: # TODO: factor too?
            self.defaults[mid] = default
        
        return mid

    def new_muts(self, batch_shape: tuple) -> tuple[jax.Array]:
        """
        Args:
            batch_shape the shape to prepend to all mutable shapes 
        Returns:
            muts The arrays to store mutable data
        """
        # print('new muts', batch_shape, self.mut_shapes)
        # print(batch_shape+self.mut_shapes[0], )
        muts = [jnp.zeros(batch_shape+shape) for shape in self.mut_shapes]
        # print(muts[0].shape)
        for mid, default in self.defaults.items():
            muts[mid.i] = muts[mid.i].at[...].set(default)

        return muts

# Shared container to keep track of mutating z_dyn, subclass so it can be used in other wrappers easily
@dataclass
class Box:
    muts: list[jax.Array]
    meta: Meta
    batch_shape: tuple = None

    def __post_init__(self):
        if len(self.meta.mut_shapes) > 0: # Leave batch_shape as None if there are no mutables
            N_test = len(self.meta.mut_shapes[0])
            self.batch_shape = self.muts[0].shape[:-N_test] if N_test > 0 else self.muts[0].shape

    def get_mut(self, mid, idx = ...):
        return self.muts[mid.i][idx]
    
    def set_mut(self, mid, value, idx = ...):
        self.muts[mid.i] = self.muts[mid.i].at[idx].set(value)

# TODO: could just check if not tuple instead
NO_IDX = '' # Don't want to use None as that is valid for broadcasting and ... causes it to index last

class Wrapper:
    class Iterable:
        @dataclass
        class Iterator:
            node_wrapper: Any
            node_iter: Any
            
            def __next__(self):
                value = self.node_iter.__next__()
                return self.node_wrapper.wrap(value)
        
        def __init__(self, node, box, idx=None):
            self.__dict__['node'] = node
            self.__dict__['box'] = box
            self.__dict__['idx'] = idx
        
        def wrap(self, value):
            # TODO: if support dynamic in init, support DynamicsMap here...
            if is_dataclass(type(value)):
                return Wrapper(value, self.box, is_root=False, idx=self.idx)
            elif callable(value):
                return Wrapper(value, self.box, is_root=False, idx=self.idx)
                # raise ValueError('Collections of functions not supported, may later support static functions')
            elif type(value) in [list, tuple, dict]:
                return Wrapper.Iterable(value, self.box, self.idx)
            else:
                return value
        
        def __getitem__(self, item):
            value = self.node[item]
            return self.wrap(value)
        
        def __iter__(self):
            # If dict then return standard key iterator
            if type(self.node) == dict:
                return self.node.__iter__()
            return self.Iterator(self, self.node.__iter__())

    # class Array:
    #     # i_pre = 
    #     # z_box = 

    #     def __init__(self, z_box, i_pre = None):
    #         self.z_box = z_box
    #         self.i_pre = i_pre

    #     def __getitem__(self, idx):
    #         return self.z_box.getz(idx, self.i_pre)

    #     def __setitem__(self, idx, value):
    #         self.z_box.setz(idx, value, self.i_pre)

    #     def __neg__(self): return self[...]._neg(self)
    #     def __add__(self, other): return self.aval._add(self, other)
    #     def __radd__(self, other): return self.aval._radd(self, other)
    #     def __mul__(self, other): return self.aval._mul(self, other)
    #     def __rmul__(self, other): return self.aval._rmul(self, other)
    #     def __gt__(self, other): return self.aval._gt(self, other)
    #     def __lt__(self, other): return self.aval._lt(self, other)
    #     def __bool__(self): return self.aval._bool(self)
    #     def __nonzero__(self): return self.aval._nonzero(self)
    
    def __init__(self, node, box: Box, is_root: bool = True, idx: tuple = NO_IDX):
        # Use __dict__ when initializing to avoid __setattr__
        self.__dict__['node'] = node
        self.__dict__['box'] = box
        self.__dict__['_idx'] = idx

    @property
    def idx(self):
        # Can't store ... in ._idx by default as don't want to prepend ... if user indices directly
        return ... if self._idx is NO_IDX else self._idx
    
    def __call__(self, *v, **k):
        _callable = self.node
        # print('_callable', _callable)
        if inspect.isfunction(_callable):
            return _callable(*v, **k)
        # TODO: member funcs
        # elif isinstance(_callable, tmethod):
        #     if _callable.is_member:
        #         return _callable(*v, tracept_self=self, **k)
        #     else:
        #         return _callable(*v, **k)
        # elif inspect.ismethod(_callable):
        #     return 
        elif hasattr(_callable, '__call__'):
            return getattr(type(_callable), '__call__')(self, *v, **k)
            # if isinstance(_callable.__call__, tmethod):
            #     return _callable(*v, tracept_self=self, **k)
            # else:
            #     raise ValueError('{} is a callable class but __call__ is not a tmethod'.format(_callable))
        else:
            raise ValueError('{} was called but is not a function, tmethod, or callable class'.format(_callable))

    def __getattr__(self, name):
        value = getattr(self.node, name) # Get value or function from actual z object
        if type(value) is MutableID:
            # TODO: to support in place slice assignments, have to wrap in something new
            return self.box.get_mut(value, idx=self.idx)
        elif inspect.ismethod(value): # Need to be able to override self so get from class here
            return lambda *v, _self=self, _f=getattr(type(self.node),name), **k: _f(_self, *v, **k)
        elif is_dataclass(type(value)) or callable(value):
            return Wrapper(value, self.box, is_root=False, idx=self.idx)
        # elif callable(value):
        #     return partial(value, tracept_self=self)
        elif type(value) in [list, tuple, dict]:
            return self.Iterable(value, self.box, self.idx)
        else:
            return value
    
    def __setattr__(self, name, value):
        leaf = getattr(self.node, name)
        if type(leaf) is MutableID:
            self.box.set_mut(leaf, value, idx=self.idx)
        else:
            raise ValueError('{} isn\'t mutable; all mutable states must be stored as a MutableID (generated from a Mutable)'.format(name))

    def __getitem__(self, idx):
        if type(idx) is not tuple: idx = (idx,)
        if type(idx[0]) is str: # Labeled access
            # if len(idx) != 2 or type(idx[1]) is not int:
            #     raise ValueError('Only (str, int) is allowed for labeled access')
            mut_ids = self.box.meta.labeled_mut_ids[idx[0]]
            if len(idx) == 1:
                return [self.box.get_mut(mid, self.idx) for mid in mut_ids]
            elif type(idx[1]) is int:
                return self.box.get_mut(mut_ids[idx[1]], self.idx)
        else:
            if self._idx is not NO_IDX:
                idx = self._idx + idx
            return Wrapper(self.node, self.box, is_root=False, idx=idx)

    def __setitem__(self, idx, value):
        if type(idx) is not tuple: idx = (idx,)
        if type(idx[0]) is str: # Labeled access
            # if len(idx) != 2 or type(idx[1]) is not int:
            #    raise ValueError('Only (str, int) is allowed for labeled writes')
            mut_ids = self.box.meta.labeled_mut_ids[idx[0]]
            if len(idx) == 1:
                for i, mid in enumerate(mut_ids):
                    self.box.set_mut(mid, value[i], self.idx)
            elif type(idx[1]) is int:
                self.box.set_mut(mut_ids[idx[1]], value, self.idx)

    def ravel_get(self, label):
        mut_ids = self.box.meta.labeled_mut_ids[label]
        return jnp.concatenate([jnp.reshape(self.box.get_mut(mid, self.idx), self.box.batch_shape+(-1,)) for mid in mut_ids], axis=-1)

    def ravel_set(self, label, values):
        mut_ids = self.box.meta.labeled_mut_ids[label]
        ptr = 0
        for i, mid in enumerate(mut_ids):
            old_mut = self.box.get_mut(mid, self.idx)
            base_size = np.prod(self.box.meta.mut_shapes[mid.i], dtype=int)
            self.box.set_mut(mid, jnp.reshape(values[...,ptr:ptr+base_size], old_mut.shape), self.idx)
            ptr += base_size

    # TODO: repr for tree structure only, dynamic only, and static only, no children ie ...
    def __format__(self, spec):
        fields = get_fields(type(self.node))
        fields_repr = type(self.node).__name__ + '( '
        do_mut, do_leaves, do_subclasses = False, False, False
        for s in spec:
            match s:
                case 'm': do_mut = True
                case 'l': do_leaves = True
                case 't': do_subclasses = True
                case _: raise ValueError('"{s}" is not a recognized format specifier, use "t" to show tclasses, "m" to show mutables, and/or "l" to show other leaves'.format(s))

        for field in fields:
            value = getattr(self.node, field.name)
            if type(value) is MutableID:
                if do_mut:
                    fields_repr += '{}={}, '.format(field.name, np.array2string(self.box.get_mut(value, idx=self.idx), max_line_width=1000))
            elif is_dataclass(type(value)): # TODO: assumed t class, allow others?
                if do_subclasses:
                    fields_repr += ('{}={:'+spec+'}, ').format(field.name, Wrapper(value, self.box, is_root=False, idx=self.idx))
            elif field.name != '__is_baked__':
                if do_leaves:
                    fields_repr += '{}={}, '.format(field.name, value)
        return fields_repr + " )"

    def __repr__(self):
        return self.__format__('mlt')

def bake_list(node_list, meta):
    for node in node_list:
        if is_dataclass(type(node)):
            bake_branch(node, meta)
        elif isinstance(node, Mutable):
            # TODO: error if __bake__?
            # mid = meta.append(node) # can't replace if dict without further mods...
            raise TypeError('{} not supported'.format(type(node)))
        # Can be static variable, leave it alone

def bake_branch(branch, meta):
    if is_dataclass(type(branch)):
        fields = get_fields(branch)
    elif type(branch) in [list, tuple]:
        return bake_list(branch, meta)
    elif type(branch) is dict:
        return bake_list(branch.values(), meta)
    else:
        raise TypeError('Unrecognized branch type {}'.format(type(branch)))
    
    # Begin by finding all Mutables and calling any pre_bake functions
    mut_nodes = {}
    for field in fields:
        node = getattr(branch, field.name)

        # print(field.name, node)
        # print(isinstance(node, Mutable) , isinstance(field.type, Mutable) , isinstance(field.type, type) and issubclass(field.type, Mutable))
        # Ensure either the value is Mutable or the dataclass type is Mutable (via either raw type or instance as type)
        if isinstance(node, Mutable) or isinstance(field.type, Mutable) or (isinstance(field.type, type) and issubclass(field.type, Mutable)):
            desc = node if isinstance(node, Mutable) else (field.type if isinstance(field.type, Mutable) else Mutable())
            bake_func = getattr(desc, '__pre_bake__', None)
            if bake_func is not None:
                bake_func(branch, mut_nodes)
            mut_nodes[field.name] = (desc, node if not isinstance(node, Mutable) else desc.default)
    # Place MutableIDs at all Mutable fields
    for name, (desc, default) in mut_nodes.items():
        setattr(branch, name, meta.append(desc, default))
    # Bake children
    for field in fields:
        node = getattr(branch, field.name)
        if isinstance(node, Wrapper):
            # Extract baked sub branch
            sub_branch, sub_meta = node.node, node.box.meta
            setattr(branch, field.name, sub_branch)
            # Append sub branch to our meta by offsetting each sub mid and carrying over defaults
            mid_offset = len(meta.mut_shapes)
            # print('node', type(sub_branch))
            # print('before', sub_meta.defaults, mid_offset)
            for sub_field in get_fields(sub_branch):
                sub_node = getattr(sub_branch, sub_field.name)
                # print('   sub_node', sub_field.name, sub_node)
                if type(sub_node) is MutableID:
                    sub_node.i += mid_offset
                    # print(sub_node)
            meta.mut_shapes += sub_meta.mut_shapes
            # Note that the mids in sub_meta are references so the offset modified above is carried over
            for label, mut_ids in sub_meta.labeled_mut_ids.items():
                meta.labeled_mut_ids[label] = meta.labeled_mut_ids.get(label, []) + mut_ids
            meta.defaults = {**meta.defaults, **sub_meta.defaults}
            # print('after', meta.defaults)
            # bake_branch(sub_branch, meta)
        elif type(node) is not MutableID and is_dataclass(type(node)) or type(node) in [list, tuple, dict]:
            # print('BAKING CHILD NODE', type(node))
            bake_branch(node, meta)

def fresh(twp, batch_shape=()):
    if type(batch_shape) is not tuple: batch_shape = (batch_shape,)
    meta = twp.box.meta
    return Wrapper(twp.node, Box(meta.new_muts(batch_shape), meta))
