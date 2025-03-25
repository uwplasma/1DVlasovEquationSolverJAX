import jax
import jax.numpy as jnp
from jax import config, jit, block_until_ready
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
from functools import partial
import matplotlib.image as img
from jaxopt import ScipyMinimize
import pickle

#Inputs/variables
v_th_e = 0.5            # Thermal Velocity of electrons normalised to speed of light (to match JAX in cell)
Ti_o_Te=0.01               #ratio of ion to electron temperatures
L=2.0                      #Length of the box
vD_e=5.0                   #Ratio of electron drift velocity to thermal electron velocity
vD_i=0.0                   #Ratio of ion drift velocity to thermal ion velocity
n0=4.0                     #Background  number density * L
n0_e=n0
n0_I=n0
dt = 1.0                          # dt normalized to c/dx  as in JAX-in-Cell 
wavenumber_electrons = 10         # Wavenumber of electrons
wavenumber_ions = 10         # Wavenumber of lectrons
A_e=0.1          #Amplitude of the spatial pertubation 
A_i=0.1              #Amplitude of the spatial pertubation 
noise_amplitude = 0.1

rtol = 1e-4
atol = 1e-4
tol_lin=1.e-8

maxiter = 50
method = "l-bfgs-b"
use_random_distribution = True
load_initial_f0 = True
number_omegap_steps_plot = 500
number_omegap_steps_array = [281]#jnp.linspace(280, 281, 30)

for i, number_omegap_steps in enumerate(number_omegap_steps_array):
    print(f"number_omegap_steps={number_omegap_steps}")
    # number_omegap_steps=65                             #This part needs to be better tested against JAX-in-Cell 

    ####Grid parameters
    nt = 5            #number of snapshots for time tracing the solution
    Nv = 100            # Number of Velocity Points
    Nx = 98             # Number of Position Points            # Number Density

    # rtol_lin = 1e-4
    # atol_lin = 1e-4

    ####################################
    ####################################

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


    #Construct initial distributions
    if use_random_distribution:
        key = jax.random.PRNGKey(0)
        f0_e = jax.random.normal(key, (Nx+2, Nv)) * 0.01
        f0_i = jax.random.normal(key, (Nx+2, Nv)) * 0.01
    else:
        f0_e=  n0_e / jnp.sqrt(2.*jnp.pi *v_th_e**2)*(0.5 * jnp.exp(-(V_e-vD_e*v_th_e)**2 / (2.*v_th_e**2)) +0.5 * jnp.exp(-(V_e+vD_e*v_th_e)**2 / (2.*v_th_e**2))) * (1 + A_e * jnp.sin(wavenumber_electrons * 2 * jnp.pi * X / L))
        f0_i = n0_I / jnp.sqrt(2.*jnp.pi * v_th_i**2) *(0.5 * jnp.exp(-(V_i-vD_i*v_th_i)**2 / (2.*v_th_i**2)) +0.5 * jnp.exp(-(V_i+vD_i*v_th_i)**2 / (2.* v_th_i**2))) * (1 + A_i * jnp.sin(wavenumber_ions* 2 * jnp.pi * X / L))
        key = jax.random.PRNGKey(0)
        f0_e += noise_amplitude*jax.random.normal(key, f0_e.shape)
        f0_i += noise_amplitude*jax.random.normal(key, f0_i.shape)

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
    # @partial(jit, static_argnums=(0))
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
    f0=(f0_e.transpose(),f0_i.transpose())


    term = diffrax.ODETerm(vector_field)
    # Temporal discretization
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_final, nt))

    # Tolerances

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    # stepsize_controller = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol)
    # Solver

    solver = diffrax.Tsit5()

    # Define the target density field from .png image
    rho_target = 1-jnp.flipud(jnp.array(img.imread('target.png')[:,:,0],dtype=float))
    zeromean_rho_target = rho_target-jnp.mean(rho_target)
    bounded_rho_target = zeromean_rho_target/jnp.max(jnp.abs(zeromean_rho_target))
    # rho_target = 1.0 + 0.02*(rho_target-0.5)
    # normalize so average density is 1
    # rho_target /= jnp.mean(rho_target)

    @jit
    def bounded_fe_solution(f0):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t_final,
            delta_t,
            f0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=5000,
            args=(x, v_e,v_i ,delta_x, delta_v_e,delta_v_i),
            progress_meter=diffrax.TqdmProgressMeter(),
        )
        fe = jnp.array(sol.ys[0][-1])
        fi = jnp.array(sol.ys[1][-1])
        zeromean_fe = fe-jnp.mean(fe)
        zeromean_fi = fi-jnp.mean(fi)
        bounded_fe = zeromean_fe/jnp.max(jnp.abs(zeromean_fe))
        bounded_fi = zeromean_fi/jnp.max(jnp.abs(zeromean_fi))
        return bounded_fe, bounded_fi

    @jit
    def loss_function(f0):
        bounded_fe, bounded_fi = bounded_fe_solution(f0)
        return jnp.mean((bounded_fe - bounded_rho_target)**2)

    # Load the optimized f0 from the pckle file
    if (not load_initial_f0) and i==0:
        f0_opt = f0
    else:
        with open('optimized_f0.pkl', 'rb') as f: f0_opt = pickle.load(f)

    optimizer = ScipyMinimize(method=method, fun=loss_function, tol=1e-8, maxiter=maxiter, options={'disp': True}, jit=True)
    result = optimizer.run(f0_opt)
    optimized_f0 = result.params

    # Save to pickle and CSV files
    with open('optimized_f0.pkl', 'wb') as f: pickle.dump(optimized_f0, f)
    with open('initial_f0.pkl', 'wb') as f: pickle.dump(f0, f)
    np.savetxt('optimized_f0e.csv', optimized_f0[0], delimiter=',')
    np.savetxt('initial_f0e.csv'  , f0[0], delimiter=',')
    np.savetxt('optimized_f0i.csv', optimized_f0[1], delimiter=',')
    np.savetxt('initial_f0i.csv'  , f0[1], delimiter=',')
    # # Load the optimized f0 from the CSV files
    # optimized_f0e = np.loadtxt('optimized_f0e.csv', delimiter=',')
    # optimized_f0i = np.loadtxt('optimized_f0i.csv', delimiter=',')
    # optimized_f0 = (optimized_f0e, optimized_f0i)
    # # Load the optimized f0 from the pckle file
    # with open('optimized_f0.pkl', 'rb') as f: optimized_f0 = pickle.load(f)

# Plot the resulting bounded_fe and the rho_target side by side
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# Plot initial bounded_fe
axes[0, 1].set_title(r'Initial $f_e$')
im2 = axes[0, 1].imshow(f0[0], origin="lower")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("v", rotation=0)
# fig.colorbar(im2, ax=axes[0, 1])

# Plot final bounded_fe after optimization
axes[0, 2].set_title(r'Optimized Initial $f_e$')
im3 = axes[0, 2].imshow(optimized_f0[0], origin="lower")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("v", rotation=0)
# fig.colorbar(im3, ax=axes[0, 2])

# # Plot final bounded_fe after optimization
# axes[0, 3].set_title(r'Optimized Initial $f_i$')
# im3 = axes[0, 3].imshow(optimized_f0[1], origin="lower")
# axes[0, 3].set_xlabel("x")
# axes[0, 3].set_ylabel("v", rotation=0)
# # fig.colorbar(im3, ax=axes[0, 2])

# Plot rho_target
axes[1, 0].set_title(f'Target $f_e$')
im1 = axes[1, 0].imshow(bounded_rho_target, origin="lower")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("v", rotation=0)
# fig.colorbar(im1, ax=axes[1, 0])

# Plot resulting bounded_fe from the simulation with initial f0
axes[1, 1].set_title(r'$f_e$ from initial $f$')
im4 = axes[1, 1].imshow(bounded_fe_solution(f0)[0], origin="lower")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("v", rotation=0)
# fig.colorbar(im4, ax=axes[1, 1])

# Plot resulting bounded_fe from the simulation with optimized f0
axes[1, 2].set_title(r'$f_e$ from optimized initial $f$')
im5 = axes[1, 2].imshow(bounded_fe_solution(optimized_f0)[0], origin="lower")
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("v", rotation=0)
# fig.colorbar(im5, ax=axes[1, 2])

# # Plot resulting bounded_fe from the simulation with optimized f0
# axes[1, 3].set_title(r'$f_i$ from optimized initial $f$')
# im5 = axes[1, 3].imshow(bounded_fe_solution(optimized_f0)[1], origin="lower")
# axes[1, 3].set_xlabel("x")
# axes[1, 3].set_ylabel("v", rotation=0)
# # fig.colorbar(im5, ax=axes[1, 2])

# Hide the empty subplot
axes[0, 0].axis('off')

plt.tight_layout()
plt.show()

# Make movie
with open('optimized_f0.pkl', 'rb') as f: f0_opt = pickle.load(f)
rho_target = 0.02*(1-jnp.flipud(jnp.array(img.imread('target.png')[:,:,0],dtype=float)))
# rho_target = (jnp.array(img.imread('target.png')[:,:,0],dtype=float)-0.5-f0_opt[0].mean())*2*jnp.abs(f0_opt[0].max())
f0_rho_target = (rho_target, f0_opt[1])

nsteps = 250
t_final = number_omegap_steps_plot/plasma_frequency
sol_rho_target = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    t_final,
    delta_t,
    f0_rho_target,
    saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t_final, nsteps)),
    stepsize_controller=stepsize_controller,
    max_steps=5000,
    args=(x, v_e,v_i ,delta_x, delta_v_e,delta_v_i),
    progress_meter=diffrax.TqdmProgressMeter(),
)
sol_optimized = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    t_final,
    delta_t,
    f0_opt,
    saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t_final, nsteps)),
    stepsize_controller=stepsize_controller,
    max_steps=5000,
    args=(x, v_e,v_i ,delta_x, delta_v_e,delta_v_i),
    progress_meter=diffrax.TqdmProgressMeter(),
)

# Scale sol_rho_target.ys[0] and sol_optimized.ys[0] to lie between 0 and 1
sol_rho_target_scaled = (sol_rho_target.ys[0] - sol_rho_target.ys[0].min()) / (sol_rho_target.ys[0].max() - sol_rho_target.ys[0].min())
sol_optimized_scaled = (sol_optimized.ys[0] - sol_optimized.ys[0].min()) / (sol_optimized.ys[0].max() - sol_optimized.ys[0].min())

# Animiation
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
im1 = axes[0,0].imshow(sol_rho_target_scaled[0], extent=(xMin, xMax, vMin_e, vMax_e), aspect='auto', cmap="jet", origin="lower")
im2 = axes[0,1].imshow(sol_optimized_scaled[ 0], extent=(xMin, xMax, vMin_e, vMax_e), aspect='auto', cmap="jet", origin="lower")
im3 = axes[1,0].imshow(sol_rho_target_scaled[0], extent=(xMin, xMax, vMin_e, vMax_e), aspect='auto', cmap="jet", origin="lower")
im4 = axes[1,1].imshow(sol_optimized_scaled[ 0], extent=(xMin, xMax, vMin_e, vMax_e), aspect='auto', cmap="jet", origin="lower")
axes[1,0].set_title(f"t = 0")
axes[1,1].set_title(f"t = 0")
cbar1 = fig.colorbar(im3, ax=axes[1, 0], label=r'$f_e$')
cbar2 = fig.colorbar(im4, ax=axes[1, 1], label=r'$f_e$')

def animate(i):
    im1.set_data(sol_rho_target_scaled[i])
    axes[0,0].set_title(f"t = {sol_rho_target.ts[i]:.2f}")
    im2.set_data(sol_optimized_scaled[i])
    axes[0,1].set_title(f"t = {sol_optimized.ts[i]:.2f}")
ani = animation.FuncAnimation(fig, animate, frames=nsteps, interval=1)
ani.save('vlasov_simulation.mp4', writer='ffmpeg', fps=30)

plt.show()
