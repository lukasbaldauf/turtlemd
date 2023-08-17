import numpy as np
import matplotlib.pyplot as plt
from turtlemd.potentials.potential import Potential
from turtlemd.potentials.jax_well import *
from turtlemd.system.box import Box
from turtlemd.system.particles import Particles
from turtlemd.system.system import System
from turtlemd.integrators import LangevinIntertia
from turtlemd.simulation import MDSimulation
from infretis.classes.engines.engineparts import write_xyz_trajectory


particles = Particles(dim=2)
box = Box(periodic=[False,False])
particles.add_particle(np.zeros(2))
particles.pos = np.array([[-0.9, 0.]])
potential = [MyWell()]
system = System(box, particles, potential)


N = 50000
simulation = MDSimulation(
    system=system, integrator=LangevinIntertia(timestep = 0.1, beta = 1.5, gamma=2.5), steps=N
)

x1 = []
x2 = []
pos = np.zeros((1,3))
vel = np.zeros((1,3))
tmp_box = [0,0,0]
atoms = ['1']
for i,step in enumerate(simulation.run()):
    if i%10==0:
        print(i/N)
        pos[:,:2] = system.particles.pos
        vel[:,:2] = system.particles.vel
        print(pos,vel, tmp_box)
        write_xyz_trajectory('traj.xyz', pos, vel,
                                atoms, tmp_box, step = i)
        x1.append(pos[0,0])
        x2.append(pos[0,1])


plt.plot(x1,x2,lw=0.5)
plt.show()
