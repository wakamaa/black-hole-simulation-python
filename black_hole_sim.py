import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Constants ---
print("--- Initializing constants ---")
c = 299792458.0  # Speed of light in m/s
G = 6.67430e-11  # Gravitational constant in N(m/kg)^2

# --- Classes (Simulating C++ structs) ---

class BlackHole:
    """
    Represents a Schwarzschild black hole.
    """
    def __init__(self, pos, mass):
        print("--- Creating BlackHole object ---")
        self.position = pos
        self.mass = mass
        self.r_s = 2.0 * G * self.mass / (c * c)
        print(f"Black hole created with mass {self.mass} kg and Schwarzschild radius {self.r_s} m")

class Ray:
    """
    Represents a photon (light ray) propagating through spacetime.
    """
    def __init__(self, pos, direction, black_hole_rs):
        print("--- Creating Ray object ---")
        # --- cartesian coords ---
        self.x = pos[0]
        self.y = pos[1]
        
        # --- polar coords ---
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.phi = np.arctan2(self.y, self.x)
        
        # --- velocities (dr/dλ, dφ/dλ) ---
        # The direction vector is in cartesian coordinates, convert to polar
        self.dr = direction[0] * np.cos(self.phi) + direction[1] * np.sin(self.phi)
        self.dphi = (-direction[0] * np.sin(self.phi) + direction[1] * np.cos(self.phi)) / self.r
        
        # --- conserved quantities ---
        f = 1.0 - black_hole_rs / self.r
        dt_dλ = np.sqrt((self.dr**2) / (f**2) + (self.r**2 * self.dphi**2) / f)
        self.E = f * dt_dλ
        self.L = self.r**2 * self.dphi
        
        # --- trail ---
        self.trail = []
        self.trail.append((self.x, self.y))
        
        print(f"Ray initialized at position ({self.x}, {self.y}) with initial polar velocities ({self.dr}, {self.dphi})")
        print(f"Conserved quantities: E={self.E}, L={self.L}")

    def step(self, d_lambda, black_hole_rs):
        """
        Integrates the ray's path forward by one step using RK4.
        """
        print(f"--- Stepping Ray at r={self.r} ---")
        if self.r <= black_hole_rs:
            print(f"Ray fell into the black hole at r={self.r}, stopping.")
            return

        # Use helper functions to perform the RK4 step
        rk4_step(self, d_lambda, black_hole_rs)
        
        # Convert back to Cartesian coordinates for visualization
        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)
        
        # Record the new point in the trail
        self.trail.append((self.x, self.y))
        print(f"Ray position updated to ({self.x}, {self.y})")

# --- Geodesic Equations and RK4 Solver ---

def geodesic_rhs(ray, rs):
    """
    Calculates the right-hand side of the geodesic equations for RK4 integration.
    This corresponds to the change in r, phi, dr/dλ, and dφ/dλ.
    """
    print("--- Calculating Geodesic RHS ---")
    r, dr, dphi, E = ray.r, ray.dr, ray.dphi, ray.E
    
    if r == 0:
        return np.zeros(4)

    f = 1.0 - rs / r
    if f == 0: # Avoid division by zero at the event horizon
        return np.zeros(4)
    
    dt_dlambda = E / f
    
    # The four derivatives we need to solve for
    # d(r)/dλ = dr
    rhs_r = dr
    # d(φ)/dλ = dφ
    rhs_phi = dphi
    
    # d(dr/dλ)/dλ from the Schwarzschild null geodesic equation
    rhs_dr = - (rs / (2 * r**2)) * f * (dt_dlambda**2) + \
             (rs / (2 * r**2 * f)) * (dr**2) + \
             (r - rs) * (dphi**2)
    
    # d(dφ/dλ)/dλ from the conservation of angular momentum
    rhs_dphi = -2.0 * dr * dphi / r
    
    rhs_values = np.array([rhs_r, rhs_phi, rhs_dr, rhs_dphi])
    print(f"RHS values calculated: {rhs_values}")
    return rhs_values

def rk4_step(ray, d_lambda, rs):
    """
    Performs a single step of the RK4 numerical integration.
    """
    print("--- Performing RK4 Step ---")
    y0 = np.array([ray.r, ray.phi, ray.dr, ray.dphi])
    print(f"Initial state y0: {y0}")

    # k1
    k1 = geodesic_rhs(ray, rs)
    
    # k2
    temp2 = y0 + (d_lambda / 2.0) * k1
    r2 = Ray((temp2[0] * np.cos(temp2[1]), temp2[0] * np.sin(temp2[1])), 
             (temp2[2], temp2[3]), rs)
    r2.r, r2.phi, r2.dr, r2.dphi = temp2[0], temp2[1], temp2[2], temp2[3]
    k2 = geodesic_rhs(r2, rs)

    # k3
    temp3 = y0 + (d_lambda / 2.0) * k2
    r3 = Ray((temp3[0] * np.cos(temp3[1]), temp3[0] * np.sin(temp3[1])),
             (temp3[2], temp3[3]), rs)
    r3.r, r3.phi, r3.dr, r3.dphi = temp3[0], temp3[1], temp3[2], temp3[3]
    k3 = geodesic_rhs(r3, rs)

    # k4
    temp4 = y0 + d_lambda * k3
    r4 = Ray((temp4[0] * np.cos(temp4[1]), temp4[0] * np.sin(temp4[1])),
             (temp4[2], temp4[3]), rs)
    r4.r, r4.phi, r4.dr, r4.dphi = temp4[0], temp4[1], temp4[2], temp4[3]
    k4 = geodesic_rhs(r4, rs)

    # Combine the k values to get the final update
    update = (d_lambda / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    print(f"Calculated update vector: {update}")
    
    ray.r += update[0]
    ray.phi += update[1]
    ray.dr += update[2]
    ray.dphi += update[3]
    print(f"New state after RK4 step: r={ray.r}, phi={ray.phi}, dr={ray.dr}, dphi={ray.dphi}")


# --- Main Simulation Loop and Visualization ---

# Setup the black hole and initial rays
SagA = BlackHole(np.array([0.0, 0.0]), 8.54e36)
rays = []
# Initialize rays from a line behind the black hole, all with velocity towards the center
for i in range(100):
    y = -1e11 + i * 2e9  # Vary the y-position
    rays.append(Ray(np.array([-3e11, y]), np.array([c, 0.0]), SagA.r_s))
    
# Setup Matplotlib plot
print("--- Setting up Matplotlib plot ---")
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor('black')
ax.set_xlim(-1.5e11, 1.5e11)
ax.set_ylim(-1.5e11, 1.5e11)
ax.set_aspect('equal')
ax.set_title('Black Hole Ray-tracing Simulation', color='white')

# Draw the black hole as a circle
black_hole_circle = plt.Circle((SagA.position[0], SagA.position[1]), SagA.r_s, color='red', fill=True)
ax.add_artist(black_hole_circle)

# Scatter plot for the rays
# Use a single scatter plot for efficiency
ray_positions = np.array([(ray.x, ray.y) for ray in rays])
scatter = ax.scatter(ray_positions[:, 0], ray_positions[:, 1], c='white', s=5)

# A list to hold the line objects for the trails
ray_trails = [ax.plot([], [], 'w-', alpha=0.5)[0] for _ in range(len(rays))]

def update(frame):
    """
    Update function for the animation.
    """
    print(f"\n--- ANIMATION FRAME {frame} ---")
    
    # Step each ray
    for ray in rays:
        ray.step(1.0, SagA.r_s)
    
    # Update the scatter plot positions
    ray_positions = np.array([(ray.x, ray.y) for ray in rays])
    scatter.set_offsets(ray_positions)
    
    # Update the trails
    for i, ray in enumerate(rays):
        if ray.trail:
            x_data, y_data = zip(*ray.trail)
            ray_trails[i].set_data(x_data, y_data)
            
    print("Plot updated.")
    return [scatter] + ray_trails

print("--- Starting animation ---")
ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)

plt.show()
