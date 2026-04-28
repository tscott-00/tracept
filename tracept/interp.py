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

# @partial(jax.jit, static_argnames='D')
def lerp(Xs: tuple[jtp.ArrayLike], X: tuple[jtp.ArrayLike], f: jtp.ArrayLike):
    if type(Xs) not in [tuple, list]: raise TypeError('Need tuple or list of Xs')

    D = len(Xs)
    I = [jnp.clip(jnp.searchsorted(X[j], Xs[j], side='right'), 1, len(X[j])-1) for j in range(D)]
    L = [jnp.clip((Xs[j] - X[j][I[j]-1]) / (X[j][I[j]] - X[j][I[j]-1]), 0.0, 1.0) for j in range(D)]
    verts = -np.array(np.unravel_index(np.arange(2**D), [2]*D)).T # (2**D, D) verts are index offsets, -1 or 0
    return reduce(lambda a,b:a+b, [f[tuple([verts[v,i] + I[i] for i in range(D)])] * reduce(lambda a,b:a*b, [1-L[d] if verts[v,d]==-1 else L[d] for d in range(D)]) for v in range(verts.shape[0])])

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
