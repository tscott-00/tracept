# Core functionality
# Authors: Thomas A. Scott https://www.scott-aero.com/

from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

# Magix classes should be initialized by user with Class.new(), which is auto-generated if not user-specified
# This is because JAX jit compiled dataclasses are internally copied by calling the constructor with all fields, preventing custom initializers
# Only variables explicitly specified in jit_statics are guaranteed to trigger recompilation
def magiclass(cls=None, *, jit_statics=[]):
    def _magiclass(cls):
        if not hasattr(cls, 'new'):
            setattr(cls, 'new', classmethod(lambda cls, *vargs, **kwargs: cls(*vargs, **kwargs)))
        jit_variables = []
        cls = dataclass(cls) # Turn into dataclass
        fields = get_fields(cls) # Get fields (everything that was annotated)
        # TODO: Error if not annotated
        for field in fields:
            if not field.name in jit_statics:
                jit_variables.append(field.name)
        # print(cls, jit_variables, jit_statics)
        jax.tree_util.register_dataclass(cls, data_fields=jit_variables, meta_fields=jit_statics)
        return cls
    
    # Handle args vs no args provided flexibility
    if cls is None:
        return _magiclass
    return _magiclass(cls)

# Class that indicates an incorrect configuration if not changed
class Placeholder:
    def __init__(self, error_message='Placeholder not set, make sure a dsp_class is constructed via MyClass.new() instead of MyClass()'):
        self.error_message = error_message

# Indicates a field will be part of the dynamic shape, user provides shape of data (at a given time for a single MC sample)
class Dynamic():
    def __init__(self, shape = (), labels=None):
        if labels == None: labels = []
        
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

class MagixWrapper:
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
                return MagixWrapper(self.z_box, value, is_root=False, i_pre=self.i_pre)
            elif callable(value):
                raise ValueError('Collections of functions not supported, may later support static functions')
            elif type(value) in [list, tuple, dict]:
                return MagixWrapper.Iterable(self.z_box, value, self.i_pre)
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
                return MagixWrapper.LerpWrapper(self.z_box, value, self.i1_pre, self.l_pre)
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
                return self.z_dyn[i_pre+(..., I)].T # TODO: better way of dealing with weird transposing np does?
                # return self.z_dyn[i_pre+(:,I,)]
        
        def setz(self, I, value, i_pre: tuple = None):
            if i_pre == None:
                self.z_dyn = self.z_dyn.at[..., I].set(value)
            else:
                self.z_dyn = self.z_dyn.at[i_pre+(..., I)].set(value)
    
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
        return self.z_node(*v, magix_self=self, **k)

    def __getattr__(self, name):
        # value = self.z_node.__dict__[name]
        value = getattr(self.z_node, name) # Get value or function from actual z object
        if type(value) is DynamicsMap:
            # TODO: to support in place slice assignments, have to wrap in something new
            return self.z_box.getz(value.I, i_pre=self.i_pre) # self.z_box.z_dyn[self.i_t,...,value.I]
        elif is_dataclass(type(value)):
            return MagixWrapper(self.z_box, value, is_root=False, i_pre=self.i_pre)
        elif callable(value):
            return partial(value, magix_self=self)
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
        return MagixWrapper(self.z_box, self.z_node, is_root=False, i_pre=i_pre)

    def lerp(self, ts: float, t: jtp.ArrayLike):
        i1 = jnp.clip(jnp.searchsorted(t, ts, side='right'), 1, len(t) - 1)
        l  = jnp.clip((ts - t[i1-1])/(t[i1] - t[i1-1]), 0.0, 1.0)
        return MagixWrapper.LerpWrapper(self.z_box, self.z_node, i1, l)

    # TODO: repr for tree structure only, dynamic only, and static only, no children ie ...
    def __repr__(self):
        fields = get_fields(type(self.z_node))
        fields_repr = ''
        for field in fields:
            value = getattr(self.z_node, field.name)
            if type(value) is DynamicsMap:
                fields_repr += '{}: {}, '.format(field.name, np.array2string(self.z_box.getz(value.I, i_pre=self.i_pre), max_line_width=1000))
            elif is_dataclass(type(value)):
                fields_repr += '{}( {} ), '.format(field.name, MagixWrapper(self.z_box, value, is_root=False, i_pre=self.i_pre))
            else:
                fields_repr += '{}: {}, '.format(field.name, value)
        return fields_repr

# Magical function decorator
# Static functions can be called from anywhere and take and return only z
# Member functions are only called from magix static functions and can take and return anything
def magixmethod(func):
    # @functools.wraps(func) # TODO: retain signature
    # Users should call with z, internal state managers like integrators should call with z_dyn and z
    def with_magix(*vargs, **kwargs):
        options = dict( # Default magix options
            z_out = True,
            # TODO: static z option
            # TODO: specific locked attribs (no read or write)
            # TODO: way to do separate containers for a set of active variables? could speed up things like pont solver a lot
        )
        if 'magix_options' in kwargs:
            for k, v in kwargs['magix_options'].items():
                options[k] = v
            del kwargs['magix_options']

        # TODO: actually detect self via inspect
        if len(vargs) > 0:
            if not 'magix_self' in kwargs:
                # TODO: could maybe allow if it returns z and can figure out where it is inside z...
                raise ValueError('Magix member functions must be called from within a magix static function as part of wrapped z')
            # TODO: allow any return?
            # if 'z' in kwargs:
                # return func(kwargs['magix_self'], *vargs[1:], z=kwargs['z'])
            # else:
                # return func(kwargs['magix_self'], *vargs[1:])
            filtered_kwargs = {k: v for k,v in kwargs.items() if k != 'magix_self'}
            # print([type(v) for v in vargs])
            return func(kwargs['magix_self'], *vargs[1:], **filtered_kwargs)
        else:
            if 'z_tree' in kwargs:
                result = func(z=MagixWrapper(kwargs['z_dyn'], kwargs['z_tree'], is_root=True))
                # TODO: allow other return values along with z?
                if options['z_out']:
                    result = result.z_box.z_dyn
            elif 'z' in kwargs:
                # If given a MagixWrapper, know this is magiception and don't intervene
                # TODO: allow generic return if nested static? take root flag for clarity?
                result = func(z=kwargs['z'])
            else:
                raise ValueError('Magix methods must be called with either wrapped z kwarg or both z_tree and z_dyn kwargs')
            return result
    
    return with_magix

def bake_list(z_list, z_ptr, dmap_z_I, dmap_dz_I, labels_I):
    for z_item in z_list:
        if is_dataclass(type(z_item)):
            z_ptr, dmap_z_I, dmap_dz_I = bake_branch(z_item, z_ptr, dmap_z_I, dmap_dz_I, labels_I)
        elif type(z_item) is Derivative:
            raise TypeError('not supported')
        elif is_dataclass(type(z_item)):
            raise TypeError('not supported')
        elif type(z_item) is Placeholder:
            raise ValueError('Field {} of {} was unset, stored error message: {}'.format(field.name, type(z_branch), value.error_message))
        # Can be static variable, leave it alone
    
    return z_ptr, dmap_z_I, dmap_dz_I, labels_I

def add_label_I(label, I, labels_I):
    if label not in labels_I:
        labels_I[label] = []
    labels_I[label].append(I)
    
    return labels_I

def bake_branch(z_branch, z_ptr, dmap_z_I, dmap_dz_I, labels_I):
    if is_dataclass(type(z_branch)):
        fields = get_fields(z_branch)
    elif type(z_branch) is list or type(z_branch) is tuple:
        return bake_list(z_branch, z_ptr, dmap_z_I, dmap_dz_I, labels_I)
    elif type(z_branch) is dict:
        return bake_list(z_branch.values(), z_ptr, dmap_z_I, dmap_dz_I, labels_I)
    else:
        raise TypeError('Unrecognized z branch type {}'.format(type(z_branch)))
    
    dmap = { }
    for field in fields:
        z_item = getattr(z_branch, field.name)
        # print(field.name, value)
        if type(z_item) is Dynamic:
            # print('Setting dynamic', field.name)
            I = z_ptr + jnp.arange(z_item.N).reshape(z_item.shape)
            for label in z_item.labels: # Record indices to dynamic variable with its labels
                add_label_I(label, I, labels_I)
            setattr(z_branch, field.name, DynamicsMap(I))
            z_ptr += z_item.N
        elif type(z_item) is Derivative:
            dmap[z_item.field_name] = field.name # map is from a fields's name to it's derivative's name
        elif is_dataclass(type(z_item)) or type(z_item) in [list, tuple, dict]:
            # print('Processing child branch', field.name)
            z_ptr, dmap_z_I, dmap_dz_I, labels_I = bake_branch(z_item, z_ptr, dmap_z_I, dmap_dz_I, labels_I)
        elif type(z_item) is Placeholder:
            raise ValueError('Field {} of {} was unset, stored error message: {}'.format(field.name, type(z_branch), z_item.error_message))
    
    # Set derivative's dynamic maps last, assumes higher order derivatives were declared in accending order
    for value_name, deriv_name in dmap.items():
        value_zmap = getattr(z_branch, value_name)
        if not type(value_zmap) is DynamicsMap:
            raise ValueError('Derivative variable {} in {} points to a {}, which should have started as a Dynamic or Derivative field and now be a DynamicsMap; if it is a derivative of a derivative, it should be declared after'.format(deriv_name, value_name, type(value_zmap)))
        I = z_ptr + jnp.arange(value_zmap.I.size).reshape(value_zmap.I.shape)
        deriv_item = getattr(z_branch, deriv_name)
        for label in deriv_item.labels: # Record indices to dynamic variable with its labels
            add_label_I(label, I, labels_I)
        setattr(z_branch, deriv_name, DynamicsMap(I))
        z_ptr += value_zmap.I.size
    
    dmap_z_I  += [getattr(z_branch,value_name).I.ravel() for value_name in dmap.keys()]
    dmap_dz_I += [getattr(z_branch,deriv_name).I.ravel() for deriv_name in dmap.values()]
    
    return z_ptr, dmap_z_I, dmap_dz_I, labels_I

# TODO: should we really be modifying z_tree?
def bake_tree(z_tree):
    """
    This WILL MODIFY z_tree
    User of lables_I will see a list of jax index arrays, i.e. separate variables, without their names
    """
    z_ptr, dmap_z_I, dmap_dz_I, labels_I = bake_branch(z_tree, z_ptr=0, dmap_z_I=[], dmap_dz_I=[], labels_I={})
    
    # Create arrays to be used in time stepping like z_dyn[...,dmap_z_I] += dt*z_dyn[...,dmap_dz_I])
    if len(dmap_z_I) > 0:
        dmap_z_I  = jnp.concatenate(dmap_z_I)
        dmap_dz_I = jnp.concatenate(dmap_dz_I)
    else:
        dmap_z_I  = jnp.zeros(0)
        dmap_dz_I = jnp.zeros(0)
    # for label, label_I in labels_I.items():
        # labels_I[label] = jnp.concatenate(label_I)

    return { 'z_tree': z_tree, 'N_dyn': z_ptr, 'dmap_z_I': dmap_z_I, 'dmap_dz_I': dmap_dz_I, 'labels_I': labels_I }

# Preprocess component classes into an aggregate z usable in magix functions, create shared dynamic array while filling z with its index maps, and resolve derivative relationships
def bake_trees(**z_branches):
    return bake_tree(magiclass(make_dataclass('_GeneratedZ', [subclass_name for subclass_name in z_branches]))(**z_branches))

def zeros(z_meta, shape=()):
    if type(shape) is int:
        shape = (shape,)
    return MagixWrapper(jnp.zeros(shape+(z_meta['N_dyn'],)), z_meta['z_tree'], is_root=True)
