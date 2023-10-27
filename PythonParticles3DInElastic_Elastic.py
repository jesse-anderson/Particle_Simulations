import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# positions are in meters (m), velocities are in meters per second (m/s), masses are in kilograms (kg), energy is in joules (J)
def check_collision(positions, velocities, masses, restitution, radius=0.1):
    energy_loss = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < 2 * radius:
                m_A = masses[i]
                m_B = masses[j]
                v_A = velocities[i]
                v_B = velocities[j]

                initial_kinetic_energy = 0.5 * m_A * np.linalg.norm(v_A)**2 + 0.5 * m_B * np.linalg.norm(v_B)**2
                
                v1_new = v_A - (2 * m_B / (m_A + m_B)) * ((v_A - v_B) @ (positions[i] - positions[j])) / np.linalg.norm(positions[i] - positions[j])**2 * (positions[i] - positions[j])
                v2_new = v_B - (2 * m_A / (m_B + m_A)) * ((v_B - v_A) @ (positions[j] - positions[i])) / np.linalg.norm(positions[j] - positions[i])**2 * (positions[j] - positions[i])

                v1_new *= restitution
                v2_new *= restitution
                
                final_kinetic_energy = 0.5 * m_A * np.linalg.norm(v1_new)**2 + 0.5 * m_B * np.linalg.norm(v2_new)**2
                energy_loss += initial_kinetic_energy - final_kinetic_energy

                velocities[i] = v1_new
                velocities[j] = v2_new
                
    return energy_loss
# Time is in seconds (s), lengths are in meters (m), diffusion coefficients are in square meters per second (m^2/s)
def simulate_diffusion_3D(num_particles, num_steps, dt, D1, D2, L, masses, restitution):
    positions = L * np.random.rand(num_particles, 3) - L / 2
    velocities = np.zeros((num_particles, 3))

    half_n = num_particles // 2

    total_energy_loss = 0

    for step in range(num_steps):
        for idx in range(num_particles):
            D = D1 if idx < half_n else D2
            dx = np.sqrt(2 * D * dt) * np.random.randn()
            dy = np.sqrt(2 * D * dt) * np.random.randn()
            dz = np.sqrt(2 * D * dt) * np.random.randn()

            dr = np.array([dx, dy, dz])
            positions[idx] += dr
            velocities[idx] = dr / dt

            positions[idx] = np.mod(positions[idx] + L / 2, L) - L / 2

        energy_loss = check_collision(positions, velocities, masses, restitution)
        total_energy_loss += energy_loss

        fig = plt.figure(figsize=(5, 5), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=4)
        ax.set_xlim(-L/2, L/2)
        ax.set_ylim(-L/2, L/2)
        ax.set_zlim(-L/2, L/2)
        ax.set_title('Particle Diffusion in 3D')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.savefig(f'frames/frame_{step}.png')
        plt.close(fig)

    print(f"Total energy loss during simulation: {total_energy_loss}")

if not os.path.exists('frames'):
    os.makedirs('frames')

num_particles = 1000
num_steps = 100
dt = 0.1
D1 = 1.0
D2 = 0.5
L = 10.0
masses = np.ones(num_particles)
restitution = 0.9

simulate_diffusion_3D(num_particles, num_steps, dt, D1, D2, L, masses, restitution)

frames = []
for i in range(num_steps):
    frames.append(imageio.imread(f'frames/frame_{i}.png'))
imageio.mimsave('particle_diffusion_3D_two_diffusivities.gif', frames, fps=10)
