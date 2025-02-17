import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import diffrax 

# Variables
vMin, vMax = -10, 10  # Velocity Range
xMin, xMax = -10, 10   # Position in x Range
Nv = 500              # Number of Velocity Points
Nx = 500              # Number of Position Points
vT = 1.0              # Thermal Velocity
n0 = 2.0              # Number Density
E0 = -1                # Constant Electric Field

v = jnp.linspace(vMin, vMax, Nv)  # Velocity Array
x = jnp.linspace(xMin, xMax, Nx)   # Position Array
delta_x = (xMax - xMin) / (Nx - 1)  # delta_x
delta_v = (vMax - vMin) / (Nv - 1)  # delta_v

# Create meshgrids for x and v and initial condition
X, V = jnp.meshgrid(x, v, indexing='ij')
f0 = n0 / jnp.sqrt(2 * jnp.pi * vT**2) * jnp.exp(-(X**2 + V**2) / (2 * vT**2))

# Central Difference in x
def central_diff_x(f, dx):
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)

# Central Difference in v
def central_diff_v(f, dv):
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dv)

# Vlasov Equation
def vector_field(t, f, args):
    v, E0, delta_x, delta_v = args
    return v * central_diff_x(f, delta_x) + E0 * central_diff_v(f, delta_v)

term = diffrax.ODETerm(vector_field)

# Temporal discretization
t0 = 0
tFinal = 1
δt = 0.0001
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, tFinal, 50))

# Tolerances
rtol = 1e-10
atol = 1e-10
stepsize_controller = diffrax.PIDController(
    pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
)

# Solver
solver = diffrax.Tsit5()
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    tFinal,
    δt,
    f0,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=None,
    args=(v, E0, delta_x, delta_v),
)

# Animiation
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(sol.ys[0], origin="lower", extent=(xMin, xMax, vMin, vMax), aspect='auto', cmap="inferno")
ax.set_xlabel("x")
ax.set_ylabel("v", rotation=0)

def animate(i):
    im.set_data(sol.ys[i])
    ax.set_title(f"t = {sol.ts[i]:.2f}")
    im.set_clim(0, 1)

ani = animation.FuncAnimation(fig, animate, frames=len(sol.ts), interval=100)

plt.show()
