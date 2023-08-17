import jax.numpy as jnp
import numpy as np
from jax import grad, jit

# float64 precision to avoid dihedral instability
from jax.config import config

from turtlemd.potentials.potential import Potential

config.update("jax_enable_x64", True)

@jit
def VMyWell(x,y):
    """
    Potential from http://dx.doi.org/10.1063/1.3029696
    """
    V = -4*jnp.exp(-0.25*(x+4)**2-y**2) \
        -4*jnp.exp(-0.25*(x-4)**2-y**2) \
        + 1/5625*(0.0425*x**6+0.5*(y-2)**6) \
        + 5*jnp.exp(-4*x**2-0.01*(y+1)**4) \
        + 5*jnp.exp(-0.0081*x**4-4*y**2) \
        - 2*jnp.exp(-20.25*((x+3)**2+(y-4.8)**2)) \
        - 2*jnp.exp(-20.25*((x+0.5)**2 + (y-3.2)**2)) 
    return V


FMyWell = jit(grad(VMyWell, argnums=(0,1)))

class MyWell(Potential):
    def __init__(
        self,
        desc="""Multistate well potential"""
        ):
        """Initialise"""
        super().__init__(desc=desc)

    def potential(self, system):
        pos = system.particles.pos
        pot = VMyWell(pos[0,0], pos[0,1])
        return pot

    def force(self, system):
        pos = system.particles.pos
        force = np.zeros(pos.shape)
        force[0,:] = FMyWell(pos[0,0], pos[0,1]) 
        force = -force
        return force, 0.0
