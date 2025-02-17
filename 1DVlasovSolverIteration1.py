import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import diffrax 

# Variables
vT = 1.0              # Thermal Velocity
vMin, vMax = -4*vT, 4*vT  # Velocity Range
xMin, xMax = -1, 1   # Position in x Range
Nv = 150              # Number of Velocity Points
Nx = 150              # Number of Position Points
n0 = 2.0              # Number Density
E0 = 0#-4              # Constant Electric Field
wavenumber = 4         # Wavenumber
t0 = 0
tFinal = 1
nt = 100
δt = 0.01
rtol = 1e-5
atol = 1e-5

v = jnp.linspace(vMin, vMax, Nv)  # Velocity Array
x = jnp.linspace(xMin, xMax, Nx)   # Position Array
delta_x = (xMax - xMin) / (Nx - 1)  # delta_x
delta_v = (vMax - vMin) / (Nv - 1)  # delta_v

# Create meshgrids for x and v and initial condition
X, V = jnp.meshgrid(x, v, indexing='ij')
f0 = n0 / jnp.sqrt(2 * jnp.pi * vT**2) * jnp.exp(-(V**2) / (2 * vT**2)) * (1 + 0.1 * jnp.sin(wavenumber * 2 * jnp.pi * X / (xMax - xMin)))

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
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, tFinal, nt))

# Tolerances
stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)#diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001)

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
im = ax.imshow(sol.ys[0].transpose(), origin="lower", extent=(xMin, xMax, vMin, vMax), aspect='auto', cmap="inferno")
ax.set_xlabel("x")
ax.set_ylabel("v", rotation=0)

def animate(i):
    im.set_data(sol.ys[i].transpose())
    ax.set_title(f"t = {sol.ts[i]:.2f}")
    im.set_clim(0, 1)

ani = animation.FuncAnimation(fig, animate, frames=len(sol.ts), interval=100)

plt.show()
