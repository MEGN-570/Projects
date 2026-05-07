# fe_electrowinning.py

"========= IMPORT MODULES ========="
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import font_manager
import numpy as np
from scipy.integrate import solve_ivp

# Plot formatting
font = font_manager.FontProperties(family='Arial', style='normal', size=12)
cmap = colormaps['plasma']


"========= LOAD INPUTS AND PARAMETERS ========="
class params:
    # Physical constants
    R = 8.314        # [J/mol-K]
    F = 96485        # [C/mol]

    # Operating conditions
    T = 298          # [K]
    i_ext = 100      # [A/m^2] applied current density (galvanostatic)

    # Electrochemistry
    n = 2            # electrons transferred
    alpha = 0.5      # symmetry factor
    i0 = 1e-3        # [A/m^2] exchange current density

    # Transport
    D = 1e-9         # [m^2/s] diffusivity of Fe2+
    c_bulk = 1000    # [mol/m^3] bulk concentration

    # Domain
    L = 1e-3         # [m] electrolyte thickness
    Nx = 50          # number of grid points


# Derived quantities
dx = params.L / (params.Nx - 1)
x = np.linspace(0, params.L, params.Nx)

nvars = params.Nx   # number of state variables (concentration at each node)


"========= INITIALIZE MODEL ========="
SV_0 = np.ones(nvars) * params.c_bulk   # initial concentration profile


"========= BUTLER-VOLMER ========="
def butler_volmer(i0, eta, p):
    return -i0 * np.exp(-p.alpha * p.n * p.F * eta / (p.R * p.T))


"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(t, SV, params):
    dSV_dt = np.zeros_like(SV)
    c = SV

    # --- Interior nodes: diffusion ---
    for i in range(1, params.Nx - 1):
        dSV_dt[i] = params.D * (c[i+1] - 2*c[i] + c[i-1]) / dx**2

    # --- Boundary: bulk electrolyte (Dirichlet) ---
    dSV_dt[-1] = 0.0  # concentration fixed at bulk

    # --- Boundary: cathode (flux from current) ---
    flux = params.i_ext / (params.n * params.F)   # mol/m^2/s

    # Fick's law: -D dc/dx = flux
    dcdx = -flux / params.D

    # ghost node method
    c_ghost = c[1] + dcdx * dx

    dSV_dt[0] = params.D * (c[1] - 2*c[0] + c_ghost) / dx**2

    return dSV_dt


"========= RUN / INTEGRATE MODEL ========="
solution = solve_ivp(
    derivative,
    [0, 10],                 # time span [s]
    SV_0,
    args=(params,),
    method='BDF'             # stiff solver (important)
)


"========= POST-PROCESSING ========="
c = solution.y

# Surface concentration (at cathode)
c_surface = c[0, :]

# Overpotential from Butler-Volmer (galvanostatic rearrangement)
eta = -(params.R * params.T / (params.alpha * params.n * params.F)) * \
      np.log(params.i_ext / params.i0)


"========= PLOTTING ========="
fig, ax = plt.subplots()
fig.set_size_inches((4,3))

# Plot concentration profiles at selected times
times_to_plot = [0, int(len(solution.t)/4), int(len(solution.t)/2), -1]

for i in times_to_plot:
    ax.plot(x*1e3, c[:, i], label=f't = {solution.t[i]:.2f} s')

ax.set_xlabel('Position (mm)')
ax.set_ylabel('Fe$^{2+}$ Concentration (mol/m$^3$)')
ax.legend(prop=font, frameon=False)
fig.tight_layout()

plt.show()


"========= OPTIONAL: SURFACE CONCENTRATION VS TIME ========="
fig2, ax2 = plt.subplots()
fig2.set_size_inches((4,3))

ax2.plot(solution.t, c_surface)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Surface Concentration (mol/m$^3$)')

fig2.tight_layout()
plt.show()