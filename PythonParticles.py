import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import imageio

def simulate_diffusion(num_particles, num_steps, dt, D, L):
    # Initialize positions
    positions = L * np.random.rand(num_particles, 2) - L / 2

    # Initialize list to store frames
    frames = []

    # Loop over time steps to update positions
    for step in range(num_steps):
        dx = np.sqrt(2 * D * dt) * np.random.randn(num_particles)
        dy = np.sqrt(2 * D * dt) * np.random.randn(num_particles)
        positions += np.column_stack([dx, dy])

        # Apply periodic boundary conditions
        positions = np.mod(positions + L / 2, L) - L / 2

        # Create a plot for the current time step
        fig, ax = plt.subplots()
        ax.scatter(positions[:, 0], positions[:, 1], s=4)
        ax.set_xlim(-L/2, L/2)
        ax.set_ylim(-L/2, L/2)
        ax.set_title('Particle Diffusion')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Convert plot to image and store
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)
    
    return frames

def update_gif():
    # Get parameters from GUI
    num_particles = int(num_particles_entry.get())
    num_steps = int(num_steps_entry.get())
    dt = float(dt_entry.get())
    D = float(D_entry.get())
    L = float(L_entry.get())

    # Run the simulation
    frames = simulate_diffusion(num_particles, num_steps, dt, D, L)

    # Create the GIF
    imageio.mimsave('particle_diffusion.gif', frames, fps=5)

# Create the Tkinter window
window = tk.Tk()
window.title("Particle Diffusion Simulator")

# Add widgets to get simulation parameters
ttk.Label(window, text="Number of Particles:").grid(column=0, row=0)
num_particles_entry = ttk.Entry(window)
num_particles_entry.grid(column=1, row=0)
num_particles_entry.insert(0, "1000")

ttk.Label(window, text="Number of Steps:").grid(column=0, row=1)
num_steps_entry = ttk.Entry(window)
num_steps_entry.grid(column=1, row=1)
num_steps_entry.insert(0, "100")

ttk.Label(window, text="Time Step (dt):").grid(column=0, row=2)
dt_entry = ttk.Entry(window)
dt_entry.grid(column=1, row=2)
dt_entry.insert(0, "0.1")

ttk.Label(window, text="Diffusion Coefficient (D):").grid(column=0, row=3)
D_entry = ttk.Entry(window)
D_entry.grid(column=1, row=3)
D_entry.insert(0, "1.0")

ttk.Label(window, text="Box Size (L):").grid(column=0, row=4)
L_entry = ttk.Entry(window)
L_entry.grid(column=1, row=4)
L_entry.insert(0, "10.0")

# Add button to run simulation and update GIF
ttk.Button(window, text="Generate GIF", command=update_gif).grid(column=0, row=5, columnspan=2)

# Run the Tkinter event loop
window.mainloop()
