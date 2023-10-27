import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Create a directory to store frames
if not os.path.exists('frames'):
    os.makedirs('frames')

def simulate_diffusion_3D(num_particles, num_steps, dt, D, L):
    # Initialize positions in 3D
    positions = L * np.random.rand(num_particles, 3) - L / 2

    # Loop over time steps to update positions
    for step in range(num_steps):
        dx = np.sqrt(2 * D * dt) * np.random.randn(num_particles)
        dy = np.sqrt(2 * D * dt) * np.random.randn(num_particles)
        dz = np.sqrt(2 * D * dt) * np.random.randn(num_particles)
        positions += np.column_stack([dx, dy, dz])

        # Apply periodic boundary conditions
        positions = np.mod(positions + L / 2, L) - L / 2

        # Create a 3D plot for the current time step
        fig = plt.figure(figsize=(10,10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=10)
        ax.set_xlim(-L/2, L/2)
        ax.set_ylim(-L/2, L/2)
        ax.set_zlim(-L/2, L/2)
        ax.set_title('Particle Diffusion in 3D')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Save the current frame
        plt.savefig(f'frames/frame_{step}.png')
        plt.close(fig)

# Define default parameters for the simulation
num_particles = 10
num_steps = 1000
dt = 0.1
D = 20
L = 100

# Run the simulation
simulate_diffusion_3D(num_particles, num_steps, dt, D, L)

# Create GIF from saved frames
frames = []
for i in range(num_steps):
    frames.append(imageio.imread(f'frames/frame_{i}.png'))
imageio.mimsave('particle_diffusion_3D.gif', frames, fps=10)
