import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Create a directory to store frames
if not os.path.exists('frames'):
    os.makedirs('frames')
# positions are in meters (m), velocities are in meters per second (m/s), masses are in kilograms (kg), energy is in joules (J)
def check_collision(positions, velocities, radius=0.1):
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < 2 * radius:
                # Collision detected, handle as perfectly elastic
                v1_new = velocities[i] - 2 * ((velocities[i] - velocities[j]) @ (positions[i] - positions[j])) / np.linalg.norm(positions[i] - positions[j])**2 * (positions[i] - positions[j])
                v2_new = velocities[j] - 2 * ((velocities[j] - velocities[i]) @ (positions[j] - positions[i])) / np.linalg.norm(positions[j] - positions[i])**2 * (positions[j] - positions[i])
                
                velocities[i] = v1_new
                velocities[j] = v2_new

# Time is in seconds (s), lengths are in meters (m), diffusion coefficients are in square meters per second (m^2/s)
def simulate_diffusion_3D(num_particles, num_steps, dt, D1, D2, L):
    # Initialize positions in 3D
    positions = L * np.random.rand(num_particles, 3) - L / 2
    velocities = np.zeros_like(positions)  # Initialize with zero velocities
    
    # Split the particles into two groups with different diffusivities
    half_n = num_particles // 2
    positions1 = positions[:half_n]
    positions2 = positions[half_n:]
    
    velocities1 = velocities[:half_n]
    velocities2 = velocities[half_n:]

    # Loop over time steps to update positions
    for step in range(num_steps):
        for positions, velocities, D in [(positions1, velocities1, D1), (positions2, velocities2, D2)]:
            dx = np.sqrt(2 * D * dt) * np.random.randn(len(positions))
            dy = np.sqrt(2 * D * dt) * np.random.randn(len(positions))
            dz = np.sqrt(2 * D * dt) * np.random.randn(len(positions))
            
            dr = np.column_stack([dx, dy, dz])
            positions += dr
            velocities = dr / dt

            # Check and handle collisions
            check_collision(positions, velocities)

            # Apply periodic boundary conditions
            positions = np.mod(positions + L / 2, L) - L / 2

        # Combine the positions for plotting
        all_positions = np.vstack([positions1, positions2])

        # Create a 3D plot for the current time step
        fig = plt.figure(figsize=(5, 5), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], s=4)
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
num_particles = 1000
num_steps = 100
dt = 0.1
D1 = 1.0  # Diffusivity for the first half
D2 = 0.5  # Diffusivity for the second half
L = 10.0

# Run the simulation
simulate_diffusion_3D(num_particles, num_steps, dt, D1, D2, L)

# Create GIF from saved frames
frames = []
for i in range(num_steps):
    frames.append(imageio.imread(f'frames/frame_{i}.png'))
imageio.mimsave('particle_diffusion_3D_two_diffusivitiesPerfectlyElastic.gif', frames, fps=10)
