"""Interpolation helpers"""

# Authors: Thomas A. Scott https://www.scott-aero.com/

from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

from tracept import Tracept, Mutable
from tracept.core import MutableID, NO_IDX

def lerp_iw(Xs: jtp.ArrayLike|tuple[jtp.ArrayLike], X: jtp.ArrayLike|tuple[jtp.ArrayLike]):
    if type(Xs) not in [tuple, list]: Xs, X = [Xs], [X] #raise TypeError('Need tuple or list of Xs')

    D = len(Xs)
    I = [jnp.clip(jnp.searchsorted(X[j], Xs[j], side='right'), 1, len(X[j])-1) for j in range(D)]
    L = [jnp.clip((Xs[j] - X[j][I[j]-1]) / (X[j][I[j]] - X[j][I[j]-1]), 0.0, 1.0) for j in range(D)]
    verts = -np.array(np.unravel_index(np.arange(2**D), [2]*D)).T # (2**D, D) verts are index offsets, -1 or 0
    # print('Xs', Xs)
    # print('X', X)
    # print('L', L)

    return [(
        tuple([verts[v,i] + I[i] for i in range(D)]),
        reduce(lambda a,b:a*b, [1-L[d] if verts[v,d]==-1 else L[d] for d in range(D)])
    ) for v in range(verts.shape[0])]

def lerp(Xs: jtp.ArrayLike|tuple[jtp.ArrayLike], X: jtp.ArrayLike|tuple[jtp.ArrayLike], f: jtp.ArrayLike):
    """
    Args:
      Xs inputs to sample at; Nx or [Nx_1, ..., Nx_D]
      X  inputs to interpolate; same shape(s) as Xs
      f  outputs to interpolate; (...)
    Returns:
      interpolated value 
    """
    iw = lerp_iw(Xs, X)
    return reduce(lambda a,b:a+b, [w*f[i] for i, w in iw])

# TODO: test
def cerp(Xs: tuple, X: tuple, f: jnp.ndarray):
    if type(Xs) not in [tuple, list]: Xs, X = [Xs], [X]

    D = len(Xs)
    I = [jnp.clip(jnp.searchsorted(X[j], Xs[j], side='right'), 1, len(X[j])-1) for j in range(D)]
    L = [jnp.clip((Xs[j] - X[j][I[j]-1]) / (X[j][I[j]] - X[j][I[j]-1]), 0.0, 1.0) for j in range(D)]
    
    # (4**D, D) verts are index offsets: -2, -1, 0, 1
    verts = np.array(np.unravel_index(np.arange(4**D), [4]*D)).T - 2 
    
    # Catmull-Rom spline basis polynomials
    def cubic_weight(t, v):
        if v == -2: return -0.5 * t**3 + t**2 - 0.5 * t
        if v == -1: return  1.5 * t**3 - 2.5 * t**2 + 1.0
        if v ==  0: return -1.5 * t**3 + 2.0 * t**2 + 0.5 * t
        if v ==  1: return  0.5 * t**3 - 0.5 * t**2

    return reduce(lambda a,b:a+b, [
        # Note the jnp.clip here to handle boundary stencils
        f[tuple([jnp.clip(verts[v,i] + I[i], 0, len(X[i])-1) for i in range(D)])] * reduce(lambda a,b:a*b, [cubic_weight(L[d], verts[v,d]) for d in range(D)]) 
        for v in range(verts.shape[0])
    ])

iw_kernels = {'lerp': lerp_iw}

class InterpWrapper:
    def __init__(self, node, box, idx: tuple, iw: list[tuple[tuple, float]]):
        self.__dict__['node'] = node
        self.__dict__['box'] = box
        self.__dict__['idx'] = idx
        self.__dict__['iw'] = iw

    def __setattr__(self, name, value):
        raise RuntimeError('Interpolation is for get access only')

    def __getattr__(self, name):
        value = getattr(self.node, name) # Get value or function from actual z object
        if type(value) is MutableID:
            m = self.box.get_mut(value,idx=self.idx)
            return reduce(lambda a,b:a+b, [w*m[i] for i, w in self.iw])
        elif isinstance(value, jax.Array):
            # print('value.shape', value.shape)
            # print('self.iw', self.iw[0])
            # print(name, value.shape)#, self.iw, value[self.idx].shape)
            return reduce(lambda a,b:a+b, [w*value[self.idx][i] for i, w in self.iw])
        elif is_dataclass(type(value)):
            return Wrapper.LerpWrapper(value, self.box, self.idx, self.iw)
        elif type(value) in [list, tuple, dict]:
            raise ValueError('Upcoming feature') # TODO: need another? or just test in wrap?
        else:
            raise ValueError(f'Can only interpolate mutables or raw jax.Array, got {type(value)}')

def interp_class(Xs: jtp.ArrayLike|tuple[jtp.ArrayLike], X: jtp.ArrayLike|tuple[jtp.ArrayLike], twp, interp_mode: str = 'lerp'):
    """On-demand interpolation across the leading dimension(s) of a Tracept object's Mutables via Tracept wrapper
    Args:
      interp_mode which interpolating type to use, e.g. 'lerp'
    Returns:
      object mimicing twp where mutable accesses will result in interpolated values
    """
    iw = iw_kernels[interp_mode](Xs, X)
    # if twp._idx is not NO_IDX:
    #     iw = [(twp._idx+i, w) for i,w in iw]
    return InterpWrapper(twp.node, twp.box, twp.idx, iw)

# # Time varing curves expressed as a linear combination of bases weighted by coeffs that may vary across MC samples
# class LerpBases:
#     x: jtp.ArrayLike # jax.scipy.interpolate.RegularGridInterpolator
#     f: jtp.ArrayLike
#     left: jtp.ArrayLike
#     right: jtp.ArrayLike
    
#     # Define number of floats needed for dynamic variables
#     coeffs: jtp.ArrayLike = Placeholder('Call LerpBases.new() to construct properly!')
    
#     @classmethod
#     def new(cls, x, f, left=None, right=None):
#         """
#         x: (Nx)
#         f: (Nx,Nf)
#         """
#         if np.any(np.isclose(np.diff(x),0)):
#             raise ValueError('x must contain unique values up to machine tolerance')
#         # if :
#         # bases = jax.scipy.interpolate.RegularGridInterpolator(x, y, method='linear')
#         return cls(x, f, left, right, jnp.ones(f.size//x.size))
    
#     @tmethod
#     def __call__(self, xs):
#         # Slop-free version of jnp.interp (edge cases are disallowed in constructor instead of handled in runtime)
#         i = jnp.clip(jnp.searchsorted(self.x, xs, side='right'), 1, len(self.x) - 1)
#         lx, = [jnp.clip((_xs - _x[_i-1])/(_x[_i] - _x[_i-1]), 0.0, 1.0) for _x, _xs, _i in [(self.x, xs, i)]]
#         # jax.debug.print('{a} {b} {c} {d} {e}', a=i, b=lx, c=len(self.x), d=self.x[i-1], e=self.x[i])
#         return self.f[i-1,...]*(1-lx) + self.f[i,...]*lx
#         # return jnp.sum(jnp.interp(xs, self.x, self.f, left=self.left, right=self.right)[None,...] * self.coeffs[...,None], axis=-1)
    
#     # @staticmethod
#     # def __class_getitem__(cls, N):
#         # return partial(self.__init__, zmap=dict(coeffs = N))

@partial(jax.tree_util.register_dataclass, data_fields=['array'], meta_fields=['inv_labels'])
@dataclass
class LabelWrapper:
    array: jtp.ArrayLike
    inv_labels: dict[str,int]
    
    def __getitem__(self, label):
        return self.array[self.inv_labels[label]]
    
    def __getattr__(self, label):
        return self.array[self.inv_labels[label]]

# TODO: kwarg constructor
# TODO: could generalize via recursive function (no overhead once compiled)
class LerpBox(metaclass=Tracept, static_attrnames=['inv_labels']):
    X: tuple[jtp.ArrayLike] # D arrays of size (Nx_i)
    f: jtp.ArrayLike # (Nx_1, ..., Nx_D, Nf)
    inv_labels: dict
    
    @property
    def D(self):
        return len(X)

    @classmethod
    def new(cls, X, f, labels=None):
        if type(X) not in [list, tuple]: X = [X]

        # Check that entries are not too close
        for d, x in enumerate(X):
            if np.any(np.isclose(np.diff(x),0)):
                raise ValueError(f'Each x in X must each contain unique values up to machine tolerance, X[{d}] does not')
        inv_labels = None if labels == None else { label: i for i, label in enumerate(labels) }

        return cls(X=X, f=f, inv_labels=inv_labels)
    
    def __call__(self, *Xs):
        fs = lerp(Xs, self.X, self.f)
        return fs if self.inv_labels == None else LabelWrapper(fs, self.inv_labels)

# # Convenience class for time varing curves the user wishes to specify as constant due to lazyness
# class DummyBases:
#     f: jtp.ArrayLike
    
#     offsets: jtp.ArrayLike = Placeholder('Call DummyBases.new() to construct properly!')
    
#     @classmethod
#     def new(cls, f, offsets=None):
#         if offsets is None:
#             offsets = jnp.zeros(f.shape)
#         return cls(f, offsets)
    
#     def __call__(self, xs):
#         return self.f
