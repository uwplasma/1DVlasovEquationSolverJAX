v_th_e = 0.7
v_th_e = 0.7            # Thermal Velocity of electrons normalised to speed of light (to match JAX in cell)
Ti_o_Te=0.01               #ratio of ion to electron temperatures
L=2.0                      #Length of the box
vD_e=5.0                   #Ratio of electron drift velocity to thermal electron velocity
vD_i=0.0                   #Ratio of ion drift velocity to thermal ion velocity
n0=1.0                     #Background  number density * L
n0_e=n0
n0_I=n0
dt = 1.0                          # dt normalized to c/dx  as in JAX-in-Cell 
wavenumber_electrons = 4.#4         # Wavenumber of electrons
wavenumber_ions = 1.         # Wavenumber of lectrons
A_e=0.001          #Amplitude of the spatial pertubation 
A_i=0.0 

nt = 300            #number of snapshots for time tracing the solution
Nv =500            # Number of Velocity Points
Nx = 500             # Number of Position Points            # Number Density

if __name__ == "__main__":
    import OneDVlasovSolverDispersion