# Interpolation helpers

from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

import magix
from magix import magiclass, Placeholder

# Time varing curves expressed as a linear combination of bases weighted by coeffs that may vary across MC samples
@magiclass#(jit_statics=['bases'])
class LerpBases:
    x: jtp.ArrayLike # jax.scipy.interpolate.RegularGridInterpolator
    f: jtp.ArrayLike
    left: jtp.ArrayLike
    right: jtp.ArrayLike
    
    # Define number of floats needed for dynamic variables
    coeffs: jtp.ArrayLike = Placeholder('Call LerpBases.new() to construct properly!')
    
    @classmethod
    def new(cls, x, f, left=None, right=None):
        """
        x: (Nx)
        f: (Nx,Nf)
        """
        if np.any(np.isclose(np.diff(x),0)):
            raise ValueError('x must contain unique values up to machine tolerance')
        # if :
        # bases = jax.scipy.interpolate.RegularGridInterpolator(x, y, method='linear')
        return cls(x, f, left, right, jnp.ones(f.size//x.size))
    
    @magixmethod
    def __call__(self, xs):
        # Slop-free version of jnp.interp (edge cases are disallowed in constructor instead of handled in runtime)
        i = jnp.clip(jnp.searchsorted(self.x, xs, side='right'), 1, len(self.x) - 1)
        lx, = [jnp.clip((_xs - _x[_i-1])/(_x[_i] - _x[_i-1]), 0.0, 1.0) for _x, _xs, _i in [(self.x, xs, i)]]
        # jax.debug.print('{a} {b} {c} {d} {e}', a=i, b=lx, c=len(self.x), d=self.x[i-1], e=self.x[i])
        return self.f[i-1,...]*(1-lx) + self.f[i,...]*lx
        # return jnp.sum(jnp.interp(xs, self.x, self.f, left=self.left, right=self.right)[None,...] * self.coeffs[...,None], axis=-1)
    
    # @staticmethod
    # def __class_getitem__(cls, N):
        # return partial(self.__init__, zmap=dict(coeffs = N))

@dataclass
class LabelWrapper:
    array: jtp.ArrayLike
    inv_labels: list[str]
    
    def __getitem__(self, label):
        return self.array[self.inv_labels[label]]
    
    def __getattr__(self, label):
        return self.array[self.inv_labels[label]]

# TODO: could generalize via recursive function (no overhead once compiled)
@magiclass(jit_statics=['inv_labels'])
class BilerpBases:
    x: jtp.ArrayLike # (Nx)
    y: jtp.ArrayLike # (Ny)
    f: jtp.ArrayLike # (Nx, Ny, Nf)
    inv_labels: dict
    
    @classmethod
    def new(cls, x, y, f, labels=None):
        # Check that entries are not too close
        if np.any(np.isclose(np.diff(x),0)) or np.any(np.isclose(np.diff(y),0)):
            raise ValueError('x and y must each contain unique values up to machine tolerance')
        if labels == None:
            inv_labels = None
        else:
            inv_labels = { label: i for i, label in enumerate(labels) }
        return cls(x, y, f, inv_labels)
    
    # TODO: allow stuff like bases(0.5)['mach'] and bases(0.5).mach?
    @magixmethod
    def __call__(self, xs, ys):
        # Direct extension of 1D version
        i, j = [jnp.clip(jnp.searchsorted(_x, _xs, side='right'), 1, len(_x)-1) for _x, _xs in [(self.x, xs), (self.y, ys)]]
        lx, ly = [jnp.clip((_xs - _x[_i-1])/(_x[_i] - _x[_i-1]), 0.0, 1.0) for _x, _xs, _i in [(self.x, xs, i), (self.y, ys, j)]]
        fs = (self.f[i-1,j-1,...]*(1-lx) + self.f[i,j-1,...]*lx)*(1-ly) + (self.f[i-1,j,...]*(1-lx) + self.f[i,j,...]*lx)*ly
        if self.inv_labels == None:
            return fs
        else:
            return LabelWrapper(fs, self.inv_labels)

# Convenience class for time varing curves the user wishes to specify as constant due to lazyness (i.e. motor mass, Cg, etc.)
@magiclass
class DummyBases:
    f: jtp.ArrayLike
    
    offsets: jtp.ArrayLike = Placeholder('Call DummyBases.new() to construct properly!')
    
    @classmethod
    def new(cls, f, offsets=None):
        if offsets is None:
            offsets = jnp.zeros(f.shape)
        return cls(f, offsets)
    
    @magixmethod
    def __call__(self, xs):
        return self.f
