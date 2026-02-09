# Core functionality
# Authors: Thomas A. Scott https://www.scott-aero.com/

from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields

import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

def tclass(cls=None, *, static_attrnames=[]):
    """Decorator to create a Tracept class, enabling functionality mutable JIT OOP.
    All attributes of the class must be annotated in dataclass convention.
    Tracept classes should be initialized by user with MyTClass.new(), which is auto-generated if not user-specified.
    This is because JAX JIT compiled dataclasses are internally copied by calling the constructor with all fields, preventing custom initializers.

    Args:
        static_attrnames: names of attributes to make static, only these attributes are guaranteed to trigger recompilation
    """
    def _tclass(cls):
        if not hasattr(cls, 'new'):
            setattr(cls, 'new', classmethod(lambda cls, *vargs, **kwargs: cls(*vargs, **kwargs)))
        jit_variables = []
        if not is_dataclass(cls):
            cls = dataclass(cls) # Turn into dataclass
        fields = get_fields(cls) # Get fields (everything that was annotated)
        # TODO: Error if not annotated
        for field in fields:
            if not field.name in static_attrnames:
                jit_variables.append(field.name)
        # print(cls, jit_variables, static_attrnames)
        jax.tree_util.register_dataclass(cls, data_fields=jit_variables, meta_fields=static_attrnames)
        return cls
    
    # Handle args vs no args provided flexibility
    if cls is None:
        return _tclass
    return _tclass(cls)

# Class that indicates an incorrect configuration if not changed
class Placeholder:
    def __init__(self, error_message='Placeholder not set, make sure a tclass is constructed via MyTClass.new() instead of MyTClass()'):
        self.error_message = error_message

# Indicates a field will be part of the dynamic shape, user provides shape of data (at a given time for a single MC sample)
class Dynamic():
    def __init__(self, default=None, shape=(), labels=None):
        """
        Args:
            default: recommended default when instantiating (e.g. used in func:fill but not func:zeros),
              should be broadcastable to arg:shape, must be broadcastable to arg:shape with z batch shape prepended
        """
        if labels == None: labels = []
        
        self.default = default
        self.shape = shape
        self.labels = labels
        if type(shape) is int:
            self.N = shape
        else: # Assume iterable, for will throw type error if shape is invalid type
            self.N = 1
            if len(shape) > 0: # Empty tuple is single value
                for N_i in shape:
                    if not type(N_i) is int:
                        raise TypeError('An iterable shape must only contain int values')
                    self.N *= N_i
                if self.N < 1:
                    raise ValueError('Must indicate at least 1 value stored')

# Indicates a derivative of a Dynamic variable
class Derivative():
    def __init__(self, field_name: str, labels=None):
        if labels == None: labels = []
        
        self.field_name = field_name
        self.labels = labels

# Dynamic and Derivative fields in a dsp_class are automatically turned into a DynamicsMap during build_z and store indices to the dynamic map
@partial(jax.tree_util.register_dataclass, data_fields=['I'], meta_fields=[])
class DynamicsMap:
    I: jtp.ArrayLike # Indices into aggregate dynamic state
    def __init__(self, I):
        self.I = I

class TWrapper:
    class Iterable:
        class Iterator:
            def __init__(self, z_node_wrapper, z_node_iter):
                self.z_node_wrapper = z_node_wrapper
                self.z_node_iter = z_node_iter
            
            def __next__(self):
                value = self.z_node_iter.__next__()
                return self.z_node_wrapper.wrap(value)
        
        def __init__(self, z_box, z_node, i_pre=None):
            self.__dict__['z_box'] = z_box
            self.__dict__['z_node'] = z_node
            self.__dict__['i_pre'] = i_pre
        
        def wrap(self, value):
            # TODO: if support dynamic in init, support DynamicsMap here...
            if is_dataclass(type(value)):
                return TWrapper(self.z_box, value, is_root=False, i_pre=self.i_pre)
            elif callable(value):
                raise ValueError('Collections of functions not supported, may later support static functions')
            elif type(value) in [list, tuple, dict]:
                return TWrapper.Iterable(self.z_box, value, self.i_pre)
            else:
                return value
        
        def __getitem__(self, item):
            value = self.z_node[item]
            return self.wrap(value)
        
        def __iter__(self):
            # print(type(self.z_node))
            # If dict then return standard key iterator
            if type(self.z_node) == dict:
                return self.z_node.__iter__()
            return self.Iterator(self, self.z_node.__iter__())
    
    class LerpWrapper:
        def __init__(self, z_dyn, z_node, i1_pre, l_pre):
            self.__dict__['z_box'] = z_dyn
            self.__dict__['z_node'] = z_node
            self.__dict__['i1_pre'] = i1_pre
            self.__dict__['l_pre'] = l_pre

        def __setattr__(self, name, value):
            raise RuntimeError('Interpolation is for get access only')

        def __getattr__(self, name):
            value = getattr(self.z_node, name) # Get value or function from actual z object
            if type(value) is DynamicsMap:
                return self.z_box.getz(value.I, i_pre=(self.i1_pre-1,))*(1-self.l_pre) + self.z_box.getz(value.I, i_pre=(self.i1_pre,))*self.l_pre
            elif is_dataclass(type(value)):
                return TWrapper.LerpWrapper(self.z_box, value, self.i1_pre, self.l_pre)
            elif type(value) in [list, tuple, dict]:
                raise ValueError('Upcoming feature') # TODO: need another? or just test in wrap?
            else:
                raise ValueError('Can only interpolate dynamics')

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

    # Shared container to keep track of mutating z_dyn, subclass so it can be used in other wrappers easily
    class ZBox:
        def __init__(self, z_dyn):
            self.z_dyn = z_dyn

        def getz(self, I, i_pre: tuple = None):
            if i_pre == None:
                return self.z_dyn[..., I]
            else:
                Dz = len(self.z_dyn.shape)
                if len(i_pre) < Dz:
                    return self.z_dyn[i_pre+(..., I)].T # TODO: better way of dealing with weird transposing np does?
                else:
                    return self.z_dyn[i_pre[:Dz-1]+(I[i_pre[Dz-1:]],)]
                # return self.z_dyn[i_pre+(:,I,)]
        
        def setz(self, I, value, i_pre: tuple = None):
            if i_pre == None:
                self.z_dyn = self.z_dyn.at[..., I].set(value)
            else:
                Dz = len(self.z_dyn.shape)
                if len(i_pre) < Dz:
                    self.z_dyn = self.z_dyn.at[i_pre+(..., I)].set(value)
                else:
                    self.z_dyn = self.z_dyn.at[i_pre[:Dz-1]+(I[i_pre[Dz-1:]],)].set(value)
    
    def __init__(self, z_dyn, z_node, is_root, i_pre=None):
        # Use __dict__ when initializing to avoid __setattr__
        if is_root:
            self.__dict__['z_box'] = self.ZBox(z_dyn)
        else:
            self.__dict__['z_box'] = z_dyn
        self.__dict__['z_node'] = z_node
        # if i_t == None:
            # i_t = slice(self.__dict__['z_box'].z_dyn.shape[0])
        if i_pre != None and type(i_pre) != tuple:
            i_pre = (i_pre,)
        self.__dict__['i_pre'] = i_pre
    
    def __call__(self, *v, **k):
        return self.z_node(*v, tracept_self=self, **k)

    def __getattr__(self, name):
        # value = self.z_node.__dict__[name]
        value = getattr(self.z_node, name) # Get value or function from actual z object
        if type(value) is DynamicsMap:
            # TODO: to support in place slice assignments, have to wrap in something new
            return self.z_box.getz(value.I, i_pre=self.i_pre) # self.z_box.z_dyn[self.i_t,...,value.I]
        elif is_dataclass(type(value)):
            return TWrapper(self.z_box, value, is_root=False, i_pre=self.i_pre)
        elif callable(value):
            return partial(value, tracept_self=self)
        elif type(value) in [list, tuple, dict]:
            return self.Iterable(self.z_box, value, self.i_pre)
        else:
            return value
    
    def __setattr__(self, name, value):
        z_leaf = self.z_node.__dict__[name]
        if type(z_leaf) is DynamicsMap:
            # self.z_box.z_dyn = self.z_box.z_dyn.at[self.i_t,...,z_leaf.I].set(value)
            self.z_box.setz(z_leaf.I, value, i_pre=self.i_pre)
        else:
            raise ValueError('{} isn\'t mutable; all mutable states must be stored as a DynamicsMap'.format(name))

    def __getitem__(self, i_pre):
        return TWrapper(self.z_box, self.z_node, is_root=False, i_pre=i_pre)

    def lerp(self, ts: float, t: jtp.ArrayLike):
        i1 = jnp.clip(jnp.searchsorted(t, ts, side='right'), 1, len(t) - 1)
        l  = jnp.clip((ts - t[i1-1])/(t[i1] - t[i1-1]), 0.0, 1.0)
        return TWrapper.LerpWrapper(self.z_box, self.z_node, i1, l)

    # TODO: repr for tree structure only, dynamic only, and static only, no children ie ...
    def __repr__(self):
        fields = get_fields(type(self.z_node))
        fields_repr = ''
        for field in fields:
            value = getattr(self.z_node, field.name)
            if type(value) is DynamicsMap:
                fields_repr += '{}: {}, '.format(field.name, np.array2string(self.z_box.getz(value.I, i_pre=self.i_pre), max_line_width=1000))
            elif is_dataclass(type(value)):
                fields_repr += '{}( {} ), '.format(field.name, TWrapper(self.z_box, value, is_root=False, i_pre=self.i_pre))
            else:
                fields_repr += '{}: {}, '.format(field.name, value)
        return fields_repr

# Tracept function decorator
# Static functions can be called from anywhere and take and return only z
# Member functions are only called from tracept static functions and can take and return anything
def tmethod(func):
    # @functools.wraps(func) # TODO: retain signature
    # Users should call with z, internal state managers like integrators should call with z_dyn and z
    def with_tracept(*vargs, **kwargs):
        options = dict( # Default tracept options
            z_out = True,
            # TODO: static z option
            # TODO: specific locked attribs (no read or write)
            # TODO: way to do separate containers for a set of active variables? could speed up things like pont solver a lot
        )
        if 'tracept_options' in kwargs:
            for k, v in kwargs['tracept_options'].items():
                options[k] = v
            del kwargs['tracept_options']

        # TODO: actually detect self via inspect
        if len(vargs) > 0:
            if not 'tracept_self' in kwargs:
                # TODO: could maybe allow if it returns z and can figure out where it is inside z...
                raise ValueError('Tracept member functions must be called from within a Tracept static function as part of wrapped z')
            # TODO: allow any return?
            # if 'z' in kwargs:
                # return func(kwargs['tracept_self'], *vargs[1:], z=kwargs['z'])
            # else:
                # return func(kwargs['tracept_self'], *vargs[1:])
            filtered_kwargs = {k: v for k,v in kwargs.items() if k != 'tracept_self'}
            # print([type(v) for v in vargs])
            return func(kwargs['tracept_self'], *vargs[1:], **filtered_kwargs)
        else:
            if 'z_tree' in kwargs:
                result = func(z=TWrapper(kwargs['z_dyn'], kwargs['z_tree'], is_root=True))
                # TODO: allow other return values along with z?
                if options['z_out']:
                    result = result.z_box.z_dyn
            elif 'z' in kwargs:
                # If given a TWrapper, know this is Traception and don't intervene
                # TODO: allow generic return if nested static? take root flag for clarity?
                result = func(z=kwargs['z'])
            else:
                raise ValueError('Tracept methods must be called with either wrapped z kwarg or both z_tree and z_dyn kwargs')
            return result
    
    return with_tracept

def bake_list(z_list, z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults):
    for z_item in z_list:
        if is_dataclass(type(z_item)):
            z_ptr, dmap_z_I, dmap_dz_I = bake_branch(z_item, z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults)
        # elif type(z_item) is Dynamic:
        #     raise TypeError('not supported')
        # elif type(z_item) is Derivative:
        #     raise TypeError('not supported')
        # elif type(z_item) is Placeholder:
        #     raise TypeError('not supported')
        else:
            raise TypeError('not supported')
            # raise ValueError('Field {} of {} was unset, stored error message: {}'.format(field.name, type(z_branch), value.error_message))
        # Can be static variable, leave it alone
    
    return z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults

def add_label_I(label, I, labels_I):
    if label not in labels_I:
        labels_I[label] = []
    labels_I[label].append(I)
    
    return labels_I

def bake_branch(z_branch, z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults):
    if is_dataclass(type(z_branch)):
        fields = get_fields(z_branch)
    elif type(z_branch) is list or type(z_branch) is tuple:
        return bake_list(z_branch, z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults)
    elif type(z_branch) is dict:
        return bake_list(z_branch.values(), z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults)
    else:
        raise TypeError('Unrecognized z branch type {}'.format(type(z_branch)))
    
    dmap = { }
    for field in fields:
        z_item = getattr(z_branch, field.name)
        # print(field.name, value)
        if type(field.type) is Dynamic or field.type is Dynamic: # Both type or instance as type are supported
            desc = z_item if type(z_item) is Dynamic else (field.type if type(field.type) is Dynamic else Dynamic())
            I = z_ptr + jnp.arange(desc.N).reshape(desc.shape)
            for label in desc.labels: # Record indices to dynamic variable with its labels
                add_label_I(label, I, labels_I)
            setattr(z_branch, field.name, DynamicsMap(I))
            default = z_item.default if (type(z_item) is Dynamic) else z_item
            if default is not None: # TODO: factor too?
                # print('DEFAULT', field.name, z_item.default)
                defaults.append((I, default))
            z_ptr += desc.N
        elif type(field.type) is Derivative:
            dmap[field.type.field_name] = field.name # map is from a fields's name to it's derivative's name
        elif is_dataclass(type(z_item)) or type(z_item) in [list, tuple, dict]:
            # print('Processing child branch', field.name)
            z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults = bake_branch(z_item, z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults)
        # if type(z_item) is Dynamic:
        #     # print('Setting dynamic', field.name)
        #     I = z_ptr + jnp.arange(z_item.N).reshape(z_item.shape)
        #     for label in z_item.labels: # Record indices to dynamic variable with its labels
        #         add_label_I(label, I, labels_I)
        #     setattr(z_branch, field.name, DynamicsMap(I))
        #     if z_item.default is not None:
        #         defaults.append((I, z_item.default))
        #     z_ptr += z_item.N
        # elif type(z_item) is Derivative:
        #     dmap[z_item.field_name] = field.name # map is from a fields's name to it's derivative's name
        # elif is_dataclass(type(z_item)) or type(z_item) in [list, tuple, dict]:
        #     # print('Processing child branch', field.name)
        #     z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults = bake_branch(z_item, z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults)
        # elif type(z_item) is Placeholder:
        #     raise ValueError('Field {} of {} was unset, stored error message: {}'.format(field.name, type(z_branch), z_item.error_message))
    
    # Set derivative's dynamic maps last, assumes higher order derivatives were declared in accending order
    for value_name, deriv_name in dmap.items():
        value_zmap = getattr(z_branch, value_name)
        if not type(value_zmap) is DynamicsMap:
            raise ValueError('Derivative variable {} in {} points to a {}, which should have started as a Dynamic or Derivative field and now be a DynamicsMap; if it is a derivative of a derivative, it should be declared after'.format(deriv_name, value_name, type(value_zmap)))
        I = z_ptr + jnp.arange(value_zmap.I.size).reshape(value_zmap.I.shape)
        # TODO: support?
        # deriv_item = getattr(z_branch, deriv_name)
        # for label in deriv_item.labels: # Record indices to dynamic variable with its labels
        #     add_label_I(label, I, labels_I)
        setattr(z_branch, deriv_name, DynamicsMap(I))
        z_ptr += value_zmap.I.size
    
    dmap_z_I  += [getattr(z_branch,value_name).I.ravel() for value_name in dmap.keys()]
    dmap_dz_I += [getattr(z_branch,deriv_name).I.ravel() for deriv_name in dmap.values()]
    
    return z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults

# TODO: should we really be modifying z_tree?
def bake_tree(z_tree):
    """
    This WILL MODIFY z_tree
    User of lables_I will see a list of jax index arrays, i.e. separate variables, without their names
    """
    z_ptr, dmap_z_I, dmap_dz_I, labels_I, defaults = bake_branch(z_tree, z_ptr=0, dmap_z_I=[], dmap_dz_I=[], labels_I={}, defaults=[])
    
    # Create arrays to be used in time stepping like z_dyn[...,dmap_z_I] += dt*z_dyn[...,dmap_dz_I])
    if len(dmap_z_I) > 0:
        dmap_z_I  = jnp.concatenate(dmap_z_I)
        dmap_dz_I = jnp.concatenate(dmap_dz_I)
    else:
        dmap_z_I  = jnp.zeros(0)
        dmap_dz_I = jnp.zeros(0)
    # for label, label_I in labels_I.items():
        # labels_I[label] = jnp.concatenate(label_I)

    return { 'z_tree': z_tree, 'N_dyn': z_ptr, 'dmap_z_I': dmap_z_I, 'dmap_dz_I': dmap_dz_I, 'labels_I': labels_I, 'defaults': defaults }

# Preprocess component classes into an aggregate z usable in Tracept functions, create shared dynamic array while filling z with its index maps, and resolve derivative relationships
def bake_trees(**z_branches):
    return bake_tree(tclass(make_dataclass('_GeneratedZ', [subclass_name for subclass_name in z_branches]))(**z_branches))

# def zeros(z_meta, shape=()):
#     """New state with all dynamic states initialized to 0.0, even if a default was specified
#     """
#     if type(shape) is int:
#         shape = (shape,)
#     return TWrapper(jnp.zeros(shape+(z_meta['N_dyn'],)), z_meta['z_tree'], is_root=True)

def fill(z_meta, shape=()):
    """New state with all dynamic states to their specified default, 0.0 for each unspecified
    """
    if type(shape) is int:
        shape = (shape,)
    z = np.zeros(shape+(z_meta['N_dyn'],))
    for I, default in z_meta['defaults']:
        z[...,I] = default
    z = jnp.array(z)

    return TWrapper(z, z_meta['z_tree'], is_root=True)
