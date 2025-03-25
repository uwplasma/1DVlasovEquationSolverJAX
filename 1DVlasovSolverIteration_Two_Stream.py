import jax
import jax.numpy as jnp
from jax import config, jit
# to use higher precision
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jax.scipy.sparse.linalg import cg
import lineax 
import numpy as np
from jax.scipy.integrate import trapezoid
from scipy.constants import elementary_charge,electron_mass,epsilon_0,Boltzmann,speed_of_light, proton_mass
import diffrax 



#Inputs/variables
v_th_e = 1.0            # Thermal Velocity of electrons normalised to speed of light (to match JAX in cell)
Ti_o_Te=0.01               #ratio of ion to electron temperatures
L=2.0                      #Length of the box
vD_e=3.0                   #Ratio of electron drift velocity to thermal electron velocity
vD_i=0.0                   #Ratio of ion drift velocity to thermal ion velocity
n0=4.0                     #Background  number density * L
n0_e=n0
n0_I=n0
dt = 1.0                          # dt normalized to c/dx  as in JAX-in-Cell 
wavenumber_electrons = 8.#4         # Wavenumber of electrons
wavenumber_ions = 1.         # Wavenumber of lectrons
A_e=0.1          #Amplitude of the spatial pertubation 
A_i=0.0              #Amplitude of the spatial pertubation 


####Grid parameters
nt = 100            #number of snapshots for time tracing the solution
Nv =100            # Number of Velocity Points
Nx = 98             # Number of Position Points            # Number Density
number_omegap_steps=2500                             #This part needs to be better tested against JAX-in-Cell 

v_th_i=v_th_e*jnp.sqrt(Ti_o_Te)*jnp.sqrt(electron_mass/proton_mass) #v_th_i normalized to c


vMin_e, vMax_e = -16*v_th_e, 16*v_th_e  # Velocity Range
vMin_i, vMax_i = -16*v_th_i, 16*v_th_i  # Velocity Range
v_e = jnp.linspace(vMin_e, vMax_e, Nv)  # Velocity Array 
v_i = jnp.linspace(vMin_i, vMax_i, Nv)  # Velocity Array
delta_x = L/ Nx   # delta_x
norm_JIC=delta_x    # To match normalization of JAX in cell together with speed of ligth normalization
xMin, xMax = -L/2.-delta_x, L/2.   # Position in x Range
delta_v_e = (vMax_e - vMin_e) / (Nv-1)  # delta_v
delta_v_i = (vMax_i - vMin_i) / (Nv-1)  # delta_v
x = jnp.linspace(-L/2.-delta_x,L/2., Nx+2)   # Position Array
delta_t=dt#*delta_x/speed_of_light  #Jax-in-Cell_definitoion, to use it we need to multiply sources by dx
charge_mass_ratio=jnp.zeros(2)  #Charge to mass ratios uin SI units
charge_mass_ratio=charge_mass_ratio.at[0].set(-elementary_charge/electron_mass) 
charge_mass_ratio=charge_mass_ratio.at[1].set(elementary_charge/proton_mass)

plasma_frequency = jnp.sqrt(n0_e * elementary_charge**2)/jnp.sqrt(electron_mass)/jnp.sqrt(epsilon_0)

#t_c=norm_JIC/speed_of_light     #normalization of time in JAX in Cell, which here is applied to the source
t0 = 0
t_final = number_omegap_steps/plasma_frequency


# Create meshgrids for x and v and initial condition, added ion mesh because thermal velocities are a lot different
X, V_i = jnp.meshgrid(x, v_i, indexing='ij')
X, V_e = jnp.meshgrid(x, v_e, indexing='ij')

rtol = 1e-7
atol = 1e-7
rtol_lin = 1e-5
atol_lin = 1e-5
tol_lin=1.e-9


#Construct initial distributions
f0_e=  n0_e / jnp.sqrt(2.*jnp.pi *v_th_e**2)*(0.5 * jnp.exp(-(V_e-vD_e*v_th_e)**2 / (2.*v_th_e**2)) +0.5 * jnp.exp(-(V_e+vD_e*v_th_e)**2 / (2.*v_th_e**2))) * (1 + A_e * jnp.sin(wavenumber_electrons * 2 * jnp.pi * X / L))
f0_i = n0_I / jnp.sqrt(2.*jnp.pi * v_th_i**2) *(0.5 * jnp.exp(-(V_i-vD_i*v_th_i)**2 / (2.*v_th_i**2)) +0.5 * jnp.exp(-(V_i+vD_i*v_th_i)**2 / (2.* v_th_i**2))) * (1 + A_i * jnp.sin(wavenumber_ions* 2 * jnp.pi * X / L))


# Central Difference in x
@jit
def central_diff_x(f, dx):
    # Get the number of grid points
    grad=(jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    #grad=grad.at[1,:].set((f.at[2,:].get()-f.at[1,:].get())/dx)
    #grad=grad.at[-2,:].set((f.at[-2,:].get()-f.at[-3,:].get())/dx)
    return grad

@jit
def central_diff_Ex(f, dx):
    # Get the number of grid points
    grad=(jnp.roll(f, -1) - jnp.roll(f, 1)) / (2 * dx)
    #grad=grad.at[1].set((f.at[2].get()-f.at[1].get())/dx)
    #grad=grad.at[-2].set((f.at[-2].get()-f.at[-3].get())/dx)
    return grad

# Central Difference in v
@jit
def central_diff_v(f, dv):
    # Get the number of grid points
    grad=(jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dv)
    #Applying f(v+delta_v)=f(v-delta_v)=0
    #grad=grad.at[:,0].set((f.at[:,1].get())/(2.*dv))
    #grad=grad.at[:,-1].set(-(f.at[:,-2].get())/(2.*dv))
    return grad

# Calculating Charge Density
@jit
def charge(f_e,f_i, v_e,v_i):
    rho = elementary_charge*(-trapezoid(f_e, x=v_e, axis=1)+ trapezoid(f_i, x=v_i, axis=1) )
    #rho = trapezoid(f_e, x=v_e, axis=1) 
    return rho 

# Calculating Potential
@jit
def potential(rho, dx):
    # Finite Differencing Matrix
    diag = -2 * jnp.ones(Nx)  # Main diagonal
    offDiag = jnp.ones(Nx-1)  # Upper and lower diagonals
    A = jnp.diag(diag) + jnp.diag(offDiag, k=1) + jnp.diag(offDiag, k=-1)
    A =jnp.true_divide(A, dx**2)
    #Adding boundaries as periodic
    A=A.at[0,-1].set( 1.)
    A=A.at[-1,0].set(1.)
    #A_lineax=lineax.MatrixLinearOperator(A)
    # Solving for the potential
    b =-rho/epsilon_0 
    solution, _ = cg(A, b, tol=tol_lin) 
    #solver=lineax.CG(rtol=rtol_lin,atol=atol_lin)
    phi=jnp.zeros(Nx+2)
    #solution=lineax.linear_solve(A_lineax,b)
    phi=phi.at[1:-1].set(solution)
    phi=phi.at[0].set(phi.at[-2].get())
    phi=phi.at[-1].set(phi.at[1].get())
    return phi

# Calculating Electric Field
@jit
def electricField(phi,dx):
    Ex=-central_diff_Ex(phi,dx)
    Ex=Ex.at[-1].set(Ex.at[1].get())
    Ex=Ex.at[0].set(Ex.at[-2].get())
    return Ex#-central_diff_x(phi,dx)


# Vlasov Equation
def vector_field(t, f, args):
    f_e,f_i=f
    #Boundaries in real space
    f_e=f_e.at[-1,:].set(f_e.at[1,:].get())
    f_i=f_i.at[-1,:].set(f_i.at[1,:].get())
    f_e=f_e.at[0,:].set(f_e.at[-2,:].get())
    f_i=f_i.at[0,:].set(f_i.at[-2,:].get())
    x, v_e,v_i, delta_x, delta_v_e,delta_v_i = args
    rho = charge(f_e[1:-1,:],f_i[1:-1,:], v_e,v_i)
    phi = potential(rho, delta_x)
    E = electricField(phi,delta_x)
    source_electrons=-norm_JIC*(V_e* central_diff_x(f_e, delta_x) +charge_mass_ratio[0]*E[:,None]* central_diff_v(f_e, delta_v_e))
    source_ions=-norm_JIC*(V_i* central_diff_x(f_i, delta_x) +charge_mass_ratio[1]*E[:,None]* central_diff_v(f_i, delta_v_i))
    source_electrons=source_electrons.at[-1,:].set(source_electrons.at[1,:].get())
    source_ions=source_ions.at[-1,:].set(source_ions.at[1,:].get())
    source_electrons=source_electrons.at[0,:].set(source_electrons.at[-2,:].get())
    source_ions=source_ions.at[0,:].set(source_ions.at[-2,:].get())    
    #jax.debug.print("electrons {t} ", t=t)
    #jax.debug.print("ions {t} ", t=source_ions)
    return source_electrons,source_ions#,E-E0


#Calculate initial electric field
rho0 = charge(f0_e[1:-1,:],f0_i[1:-1,:], v_e,v_i)
phi0 = potential(rho0, delta_x)
E0=electricField(phi0,delta_x)
f0=(f0_e,f0_i)


term = diffrax.ODETerm(vector_field)
# Temporal discretization
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_final, nt))

# Tolerances

stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
#stepsize_controller = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol)
# Solver
solver = diffrax.Tsit5()
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    t_final,
    delta_t,
    f0,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=None,
    args=(x, v_e,v_i ,delta_x, delta_v_e,delta_v_i),
    progress_meter=diffrax.TqdmProgressMeter(),
)




# Animiation electrons
fig, ax = plt.subplots(figsize=(5, 5))
plt.title('Electrons')
im = ax.imshow(sol.ys[0][0].transpose(), origin="lower", extent=(xMin, xMax, vMin_e, vMax_e), aspect='auto', cmap="inferno")
ax.set_xlabel("x")
ax.set_ylabel("v/c", rotation=0)

def animate(i):
    im.set_data(sol.ys[0][i].transpose())
    ax.set_title(f"$t\\omega_P$ = {sol.ts[i]*plasma_frequency:.2f}")
    #im.set_clim(0, 1)


ani = animation.FuncAnimation(fig, animate, frames=len(sol.ts), interval=100)

plt.show()

# # Animiation ions
# fig, ax = plt.subplots(figsize=(5, 5))
# plt.title('Ions')
# im = ax.imshow(sol.ys[1][0].transpose(), origin="lower", extent=(xMin, xMax, vMin_i, vMax_i), aspect='auto', cmap="inferno")
# ax.set_xlabel("x")
# ax.set_ylabel("v/c", rotation=0)

# def animate(i):
#     im.set_data(sol.ys[1][i].transpose())
#     ax.set_title(f"$t\\omega_P$ = {sol.ts[i]*plasma_frequency:.2f}")
#     #im.set_clim(0, 1)

# ani = animation.FuncAnimation(fig, animate, frames=len(sol.ts), interval=100)

# plt.show()
