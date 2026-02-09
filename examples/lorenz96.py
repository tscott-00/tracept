"""Ordinary differential equation integration example using the chaotic Lorenz96 system"""

# Authors: Thomas A. Scott https://www.scott-aero.com/

import os
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import tracept
from tracept import tclass, tmethod, Dynamic, Derivative
import tracept.odes

@tclass
class Lorenz96:
    # x:  jax.Array = Placeholder()
    # dx: jax.Array = Derivative('x')
    # Above is unnecessary boilerplate, for value can either be broadcastabled default or Dynamic instance to specify shape 
    # TODO: remove Placeholder
    x:  Dynamic
    dx: Derivative('x') = None

    @classmethod
    def new(cls, dims):
        """Lorenz96 dynamics for a user-specified dimensionality."""

        return cls(x=Dynamic(8.0, shape=dims))

    @tmethod
    def __call__(self):
        """Update derivatives using internal state.

        $\\frac{dx_i}{dt} = (x_{i+1} - x_{i-2}})x_{i-1} + 8$, with indices wrapping around when under or overflowing
        """

        self.dx = (jnp.roll(self.x,-1,axis=-1) - jnp.roll(self.x,2,axis=-1))*jnp.roll(self.x,1,axis=-1) - self.x + 8.0

if __name__ == "__main__":
    N = 2 # Number of distinct sims to run simulatenously
    z_meta = tracept.bake_tree(Lorenz96.new(dims=8))
    integrator = tracept.odes.make_integrator(z_meta, tracept.odes.step_fe)

    # Allocate state with state and derivatives starting at zeros then initialize x to 1.0
    z0 = tracept.fill(z_meta, shape=N)
    # z0.x = 8.0
    # Apply proturbations
    for i in range(N):
        # Note that z0 is a Tracept object but z0[...].x is a JAX array
        #   i.e. in-place operations must always be preemptively indexed (here i is batch index, 0 is index in x)
        z0[i,0].x += (i+1)*0.01
        # To emphasize the indexing point, this also works
        # z0[i,0].x = z0[i].x[0] + (i+1)*0.01
        # However, this does not
        # z0[i].x[0] += (i+1)*0.01

    # Run JIT compiled integrator
    t, z = integrator(z0, dt=1E-2, T=30.0)
    # Print state at final time
    print('Output shapes and terminal states:', z.x.shape, z[-1].x.shape)
    print(z[-1].x)
    print('Is lerp working:', np.allclose((z[0].x+z[1].x)/2, z.lerp(0.5, np.arange(t.size)).x))

    # TODO: Plot first 3 states
