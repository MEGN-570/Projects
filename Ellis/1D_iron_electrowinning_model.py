# 1D_iron_electrowinning_model.py

"========= IMPORT MODULES ========="
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import font_manager
import numpy as np
from scipy.integrate import solve_ivp

# Plotting formatting:
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=12)

ncolors = 4
ind_colors = np.linspace(0, 1.15, ncolors)
cmap = colormaps['plasma']
colors = cmap(ind_colors)


"========= LOAD INPUTS AND OTHER PARAMETERS ========="

class params:

    # Operating conditions
    i_app = 5000         # [A/m^2] Applied current density
    T = 298             # [K] Temperature

    # Electrochemistry
    n = 2               # [-] electrons transferred
    i0 = 10             # [A/m^2] exchange current density
    beta = 0.5          # [-] symmetry factor

    # Thermodynamics
    E0 = -0.44          # [V] standard potential

    # Transport
    D_Fe = 1e-9         # [m^2/s] diffusivity

    # Physical properties
    M = 0.055845        # [kg/mol]
    rho = 7870          # [kg/m^3]

    # Electrolyte conductivity
    kappa = 5           # [S/m]

    # Domain definition
    L = 1e-3            # [m] electrolyte thickness
    Nx = 50             # number of spatial nodes

    # Initial conditions
    C0 = 1000           # [mol/m^3]


# Constants
R = 8.314
F = 96485

# Spatial discretization
dx = params.L / (params.Nx - 1)
x = np.linspace(0, params.L, params.Nx)

# Number of state variables
nvars = params.Nx + 1


"========= POINTERS ========="

class ptr:

    # Concentration nodes
    C_Fe = np.arange(params.Nx)

    # Deposit thickness
    delta = params.Nx


"========= INITIALIZE MODEL ========="

SV_0 = np.zeros((nvars,))

# Initial concentration profile
SV_0[ptr.C_Fe] = params.C0

# Initial deposit thickness
SV_0[ptr.delta] = 0.0


"========= BUTLER VOLMER ========="

def butler_volmer(i0, eta, C_surface, p):
    i0_eff = i0 * (C_surface / p.C0)
    BV = i0_eff*(np.exp(-p.beta*p.n*F*eta/(R*p.T)) - np.exp((1-p.beta)*p.n*F*eta/(R*p.T)))
    
    return BV

"========= SOLVE OVERPOTENTIAL ========="

def solve_overpotential(i_target, C_surface, p):

    eta = -0.05

    for j in range(100):

        i_calc = butler_volmer(p.i0, eta, C_surface, p)

        resid = i_calc - i_target

        deta = 1e-6

        i2 = butler_volmer(p.i0, eta + deta, C_surface, p)

        dresid = (i2 - i_calc) / deta

        eta_new = eta - resid / dresid

        if abs(eta_new - eta) < 1e-8:
            break

        eta = eta_new

    return eta


"========= DEFINE RESIDUAL FUNCTION ========="

def derivative(t, SV, params, ptr):

    dSV_dt = np.zeros_like(SV)

    # Unpack variables
    C = SV[ptr.C_Fe]
    delta = SV[ptr.delta]

    # Prevent negative concentrations
    C = np.maximum(C, 1e-12)

    # Surface concentration
    C_surface = C[0]

    # ========= LIMITING CURRENT =========

    i_lim = params.n * F * params.D_Fe * C_surface / dx

    # Actual achievable current
    # i = min(params.i_app, i_lim)
    i = params.i_app*i_lim/(params.i_app + i_lim)

    # Stop simulation if depleted
    if i <= 1e-6:

        dSV_dt[:] = 0

        return dSV_dt

    # ========= ELECTROCHEMISTRY =========

    # Nernst equation
    E_eq = params.E0 + (R * params.T) / (params.n * F) * np.log(C_surface)

    # Solve Butler-Volmer equation
    eta = solve_overpotential(i, C_surface, params)

    # Butler-Volmer current
    i_BV = butler_volmer(params.i0, eta, C_surface, params)

    # Ohmic loss
    eta_ohmic = i * params.L / params.kappa

    # Cell voltage
    V_cell = E_eq + eta + eta_ohmic

    # ========= TRANSPORT =========

    # Flux boundary condition
    flux = i / (params.n * F)

    # Surface concentration gradient
    dCdx_surface = -flux / params.D_Fe

    # Ghost node
    C_ghost = C[1] + dCdx_surface * dx

    # ========= DIFFUSION =========

    # Cathode node
    dSV_dt[0] = params.D_Fe * (C[1] - 2*C[0] + C_ghost) / dx**2

    # Interior nodes
    for j in range(1, params.Nx - 1):

        dSV_dt[j] = params.D_Fe * (C[j+1] - 2*C[j] + C[j-1]) / dx**2

    # Bulk boundary condition
    dSV_dt[params.Nx - 1] = 0.0

    # ========= MOVING BOUNDARY =========

    dDeltadt = params.M * i / (params.n * F * params.rho)

    dSV_dt[ptr.delta] = dDeltadt

    return dSV_dt


"========= RUN / INTEGRATE MODEL ========="

solution = solve_ivp(

    derivative,

    [0, 2000],

    SV_0,

    args=(params, ptr),

    method='BDF',

    t_eval=np.linspace(0, 2000, 200)

)


"========= POST-PROCESSING ========="

# Extract variables
C = solution.y[ptr.C_Fe]
delta = solution.y[ptr.delta]
t = solution.t

# Surface concentration
C_surface = C[0, :]

# Deposit thickness in mm
delta_mm = delta * 1e3

# Compute limiting current history
i_lim = params.n * F * params.D_Fe * C_surface / dx

# Actual current
i_actual = np.minimum(params.i_app, i_lim)

# Voltage components
E_eq = np.zeros_like(t)
eta_act = np.zeros_like(t)
eta_ohmic = np.zeros_like(t)
V_cell = np.zeros_like(t)

for j in range(len(t)):

    # Surface concentration
    Cs = C_surface[j]

    # Equilibrium voltage
    E_eq[j] = params.E0 + (R * params.T) / (params.n * F) * np.log(Cs)

    # Activation overpotential
    eta_act[j] = solve_overpotential(i_actual[j], Cs, params)

    # Ohmic loss
    eta_ohmic[j] = i_actual[j] * params.L / params.kappa

    # Total cell voltage
    V_cell[j] = abs(E_eq[j] + eta_act[j] + eta_ohmic[j])


"========= PLOT CONCENTRATION PROFILES ========="

fig, ax = plt.subplots()

fig.set_size_inches((4,3))

plot_times = [0, 50, 100, 199]

for i, ind in enumerate(plot_times):

    ax.plot(x * 1e3,
            C[:, ind],
            color=colors[i],
            label=f't = {t[ind]:.0f} s')

ax.set_xlabel('Position [mm]')
ax.set_ylabel('Fe$^{2+}$ Concentration [mol/m$^3$]')

ax.legend(prop=font, frameon=False)

fig.tight_layout()

plt.show()


"========= PLOT SURFACE CONCENTRATION ========="

fig2, ax2 = plt.subplots()

fig2.set_size_inches((4,3))

ax2.plot(t,
         C_surface,
         color='purple')

ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Surface Concentration [mol/m$^3$]')

fig2.tight_layout()

plt.show()


"========= PLOT DEPOSIT THICKNESS ========="

fig3, ax3 = plt.subplots()

fig3.set_size_inches((4,3))

ax3.plot(t,
         delta_mm,
         color='gold')

ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Deposit Thickness [mm]')

fig3.tight_layout()

plt.show()


"========= PLOT CURRENT LIMITING ========="

fig4, ax4 = plt.subplots()

fig4.set_size_inches((4,3))

ax4.plot(t,
         i_actual,
         color='darkred',
         label='Actual Current')

ax4.axhline(params.i_app,
            color='black',
            linestyle='--',
            label='Applied Current')

ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Current Density [A/m$^2$]')

ax4.legend(prop=font, frameon=False)

fig4.tight_layout()

plt.show()


"========= PLOT CELL VOLTAGE ========="

fig5, ax5 = plt.subplots()

fig5.set_size_inches((4,3))

ax5.plot(t,
         V_cell,
         color='darkgreen')

ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Cell Voltage [V]')

fig5.tight_layout()

plt.show()


"========= SAVE FIGURES ========="

fig.savefig('concentration_profiles.png', dpi=300, bbox_inches='tight')
fig2.savefig('surface_concentration.png', dpi=300, bbox_inches='tight')
fig3.savefig('deposit_thickness.png', dpi=300, bbox_inches='tight')
fig4.savefig('current_limiting.png', dpi=300, bbox_inches='tight')
fig5.savefig('cell_voltage.png', dpi=300, bbox_inches='tight')