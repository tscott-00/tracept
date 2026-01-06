# ODE helpers
# Authors: Thomas A. Scott https://www.scott-aero.com/

from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

from magix import magixmethod
from magix.core import MagixWrapper

# Forward Euler scheme
def step_fe(
    z_dyn, z_tree, dmap_z_I, dmap_dz_I,
    dt, fmagix_dyn,
):
    return z_dyn.at[...,dmap_z_I].add(dt*z_dyn[...,dmap_dz_I])

# Generic integrator step, currently set up for predetermind time steps
def integrator_step(i, args, fstep, fmagix_dyn):
    t, z_tree, dmap_z_I, dmap_dz_I, z_dyn, z_dyn_stack = args
    dt = t[i] - t[i-1]

    # Calculate derivative at last time and record, i.e. the state at i-1
    z_dyn = fmagix_dyn(z_dyn=z_dyn, z_tree=z_tree)
    z_dyn_stack = z_dyn_stack.at[i-1,...].set(z_dyn)
    
    # Call integrator to progress independent variables from i-1 to i
    z_dyn = fstep(z_dyn, z_tree, dmap_z_I, dmap_dz_I, dt, fmagix_dyn)

    return t, z_tree, dmap_z_I, dmap_dz_I, z_dyn, z_dyn_stack

# Integrator that takes a magix dynamics function
def make_integrator(z_meta, fstep, fmagix_dyn=None):
    z_tree, dmap_z_I, dmap_dz_I = [z_meta[k] for k in ['z_tree', 'dmap_z_I', 'dmap_dz_I']]
    if fmagix_dyn == None: # When dyn not provided, it is z_tree itself which is magix 
        @magixmethod
        def fmagix_dyn(z):
            z()
            return z
        # fmagix_dyn = magixmethod(lambda z: z())
    # ODE Integrator function
    # TODO: this should be vmap or pmap outside, adjacent memory means doing each individually
    # Optimizing the NN should have batches as the inner dim (time as outer) but need to not copy s every time...

    # TODO: get deriv at time 0 for completeness? storing dx needs to be out of sync with x!
    _integrator_step = jax.jit(partial(integrator_step, fstep=fstep, fmagix_dyn=fmagix_dyn))
    
    def _integrator(z0, dt, T, _integrator_step=_integrator_step, dmap_z_I=dmap_z_I, dmap_dz_I=dmap_dz_I):
        if type(z0) is MagixWrapper:
            z_dyn0 = z0.z_box.z_dyn
            z_tree = z0.z_node
        else:
            raise TypeError('Must use a magix wrapped z to start the integrator.')

        Nt = jnp.ceil(T / dt).astype(int)
        t = (jnp.arange(Nt)*dt).at[-1].set(T)
        
        z_dyn_stack = jnp.zeros((Nt,)+z_dyn0.shape)
        # print(_integrator_step.lower(0,(t, z_tree, dmap_z_I, dmap_dz_I, z_dyn0, z_dyn_stack)).as_text())
        _, _, _, _, z_dyn, z_dyn_stack = jax.lax.fori_loop(1, Nt, _integrator_step, (t, z_tree, dmap_z_I, dmap_dz_I, z_dyn0, z_dyn_stack))
        
        # Final state in general only has independent variables at final time after exiting integrator, call dynamics once more to update to full state at final time
        z_dyn = fmagix_dyn(z_dyn=z_dyn, z_tree=z_tree)
        z_dyn_stack = z_dyn_stack.at[-1,...].set(z_dyn)
        
        return t, MagixWrapper(z_dyn_stack, z_tree, is_root=True)
    
    return _integrator
