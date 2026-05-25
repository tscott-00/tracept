"""ODE helpers"""

# Authors: Thomas A. Scott https://www.scott-aero.com/

from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

from tracept import Mutable
from tracept.core import Wrapper, Box

class Derivative(Mutable):
    """A mutable that represents the derivative of another"""
    def __init__(self, field_name: str, default=None, other_labels=[]):
        # Dynamic.__init__(self, default ['deriv']+other_labels)
        self.field_name = field_name
        self.default = default
        self.labels = ['derivs']+other_labels

    # TODO: needs to make sure it gets added to meta in same order!!!
    def __pre_bake__(self, owner, owners_mut_nodes):
        # TODO: issue with this one is will just crash 
        # Find our corresponding state
        state_ptr, corresponding_state = 0, None
        for i, (mut_name, mut_desc, _) in enumerate(owners_mut_nodes):
            if mut_name == self.field_name:
                i_corresponding_state, corresponding_state = i, mut_desc
        if corresponding_state is None: raise ValueError(f'Coud not find corresponding state "{self.field_name}", ensure it is in the same branch')
        
        # First derivative to be processesed marks all state-derivative pairs on this branch
        if 'states' not in corresponding_state.labels:
            for i, (mut_name, mut_desc, _) in enumerate(owners_mut_nodes):
                if 'derivs' in mut_desc.labels: # This is before any deriv str labels become tuples
                    for j, (other_mut_name, other_mut_desc, _) in enumerate(owners_mut_nodes):
                        if other_mut_name == mut_desc.field_name:
                            if 'states' in other_mut_desc.labels: raise ValueError(f'{self.field_name} cannot already be marked as a memeber of states when a derivative is specified')
                            other_mut_desc.labels.append('states')
                            break
        
        # Count all states proceeding corresponding_state to get its index
        for i in range(i_corresponding_state):
            mut_desc = owners_mut_nodes[i][1]
            if 'states' in mut_desc.labels:
                state_ptr += 1
        
        self.labels[0] = ('derivs', state_ptr)
        # print('placing deriv of ', self.field_name, 'at ', state_ptr)
        
        self.shape = corresponding_state.shape

# Forward Euler scheme
def step_fe(twp, dt):
    # print(twp['derivs'])
    # for i, deriv in enumerate(twp['derivs']):
    #     twp['states',i] += dt*deriv
    twp['states'] = [s + dt*ds for s, ds in zip(twp['states'], twp['derivs'])]
    # TODO: should we support twp['states'][i] += ... would need to return to return a wrapper ref list which inconiences multi steppers since need to manually copy
    #       just put labeled tutorial and make clear this does not give a reference

def update_and_record(i, tin, meta, muts, mut_stacks):
    twp = Wrapper(tin, Box(muts, meta))
    twp()
    # print(mut_stacks[0].shape, twp.box.muts[0].shape)
    mut_stacks = [mut_stack.at[i,...].set(mut) for mut_stack, mut in zip(mut_stacks, twp.box.muts)]

    return twp, mut_stacks

# Generic integrator step, currently set up for predetermind time steps
def integrator_step(i, args, fstep):
    t, tin, meta, muts, mut_stacks = args
    dt = t[i] - t[i-1]

    # Calculate derivative at last time and record, i.e. the state at i-1
    twp, mut_stacks = update_and_record(i-1, tin, meta, muts, mut_stacks)
    
    # Call integrator to progress independent variables from i-1 to i
    fstep(twp, dt)

    return t, tin, meta, twp.box.muts, mut_stacks

# Integrator that takes a Tracept dynamics function
def make_integrator(fstep):
    # z_tree, dmap_z_I, dmap_dz_I = [z_meta[k] for k in ['z_tree', 'dmap_z_I', 'dmap_dz_I']]
    # if upd_dyn == None: # When dyn not provided, it is z_tree itself which is a tclass 
    #     @tmethod
    #     def upd_dyn(z):
    #         z()
    #         return z
        # upd_dyn = tmethod(lambda z: z())
    # ODE Integrator function
    # TODO: this should be vmap or pmap outside, adjacent memory means doing each individually
    # Optimizing the NN should have batches as the inner dim (time as outer) but need to not copy s every time...

    # TODO: get deriv at time 0 for completeness? storing dx needs to be out of sync with x!
    _integrator_step = jax.jit(partial(integrator_step, fstep=fstep))
    
    def _integrator(twp0, dt, T, _integrator_step=_integrator_step):
        if type(twp0) is Wrapper:
            tin, box = twp0.node, twp0.box
            muts0, meta = box.muts, box.meta
            if len(muts0) == 0: raise TypeError('Cannot integrate a type with no mutable variables')
        else:
            raise TypeError('Must use an instance of a class created through the Tracepted metaclass.')

        Nt = jnp.ceil(T / dt).astype(int) + 1
        t = (jnp.arange(Nt)*dt).at[-1].set(T)
        
        # print('box.batch_shape', box.batch_shape, box.muts[0].shape)
        mut_stacks = meta.new_muts((Nt,)+box.batch_shape)
        # print(_integrator_step.lower(0,(t, z_tree, dmap_z_I, dmap_dz_I, z_dyn0, z_dyn_stack)).as_text())
        _, _, _, muts, mut_stacks = jax.lax.fori_loop(1, Nt, _integrator_step, (t, tin, meta, muts0, mut_stacks))
        
        # Final state in general only has independent variables at final time after exiting integrator, call dynamics once more to update to full state at final time
        _, mut_stacks = update_and_record(-1, tin, meta, muts, mut_stacks)
        
        return t, Wrapper(tin, Box(mut_stacks, meta))
    
    return _integrator
