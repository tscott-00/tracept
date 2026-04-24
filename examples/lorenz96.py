"""Ordinary differential equation integration example using the chaotic Lorenz96 system"""

# Authors: Thomas A. Scott https://www.scott-aero.com/

import os
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import tracept
from tracept import Tracepted, Mutable
from tracept.odes import Derivative

# TODO: is meta really a good idea? needs to only happen at highest level - not great for interopability
#    unless children can be wrapped - could detect if store the original tin (before replacing stuff with mid) or offsetting their mid 
#    can't modify Mutable etc between init and bake but not a big issue

class Lorenz96(metaclass=Tracepted):
    x:  Mutable
    dx: Derivative('x') = None

    @classmethod
    def new(cls, dims, batch_shape=()):
        """Lorenz96 dynamics for a user-specified dimensionality."""

        return cls(x=Mutable(8.0, shape=dims), batch_shape=batch_shape)

    # @tmethod
    def __call__(self):
        """Update derivatives using internal state.

        $\\frac{dx_i}{dt} = (x_{i+1} - x_{i-2}})x_{i-1} + 8$, with indices wrapping around when under or overflowing
        """

        self.dx = (jnp.roll(self.x,-1,axis=-1) - jnp.roll(self.x,2,axis=-1))*jnp.roll(self.x,1,axis=-1) - self.x + 8.0

if __name__ == "__main__":
    import argparse

    N = 2 # Number of distinct sims to run simulatenously
    state = Lorenz96.new(dims=8, batch_shape=N)

    # Apply proturbations
    for i in range(N):
        # Note that z0 is a Tracept object but z0[...].x is a JAX array (or slice of one)
        #   assignment of x (but not if subsequently sliced) will be intercepted, enabling in-place modificationss
        #   note that in-place operations must always be preemptively indexed (here i is batch index, 0 is index in x)
        state[i,0].x += (i+1)*0.01
        # To emphasize the indexing point, these also work
        # state[i,0].x = state[i].x[0] + (i+1)*0.01
        # state[i].x = state[i].x.at[0].add((i+1)*0.01)
        # However, this does not
        # state[i].x[0] += (i+1)*0.01

        # TODO: no longer a pitfall since no global array
        # Note the pitfall, here z0.x will not be modified, only _x
        #   this is the same behavior as numpy and regular JAX, storing a slice creates a copy not a reference
        # _x = state.x
        # _x += 1

    # Run JIT compiled integrator
    t, states = tracept.odes.make_integrator(tracept.odes.step_fe)(state, dt=1E-2, T=30.0)
    # Print state at final time
    print('Output shapes and terminal states:', states.x.shape, states[-1].x.shape)
    print(states[-1].x)
    print('Is lerp working:', np.allclose((states[0].x+states[1].x)/2, states.lerp(0.5, np.arange(t.size)).x))

    # TODO: Plot first 3 states

# TODO: need another example that uses a list, tuple, and dict
