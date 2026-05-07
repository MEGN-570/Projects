# 1D_iron_electrowinning_model.py

"========= IMPORT MODULES ========="
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import font_manager
import numpy as np
from scipy.integrate import solve_ivp


"========= LOAD INPUTS AND OTHER PARAMETERS ========="

class params:

    # Operating conditions
    # Operating conditions
    i_app_sweep = [100, 500, 1000, 2500, 5000, 10000]
    i_app = None        # [A/m^2] Applied current density
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
def run_model(i_app_value):

    params.i_app = i_app_value

    solution = solve_ivp(
        derivative,
        [0, 2000],
        SV_0,
        args=(params, ptr),
        method='BDF',
        t_eval=np.linspace(0, 2000, 200))

    # ========= POST PROCESS =========

    C = solution.y[ptr.C_Fe]
    delta = solution.y[ptr.delta]
    t = solution.t

    C_surface = C[0, :]

    delta_mm = delta * 1e3

    i_lim = params.n*F*params.D_Fe*C_surface/dx

    i_actual = params.i_app*i_lim/(params.i_app + i_lim)

    E_eq = np.zeros_like(t)
    eta_act = np.zeros_like(t)
    eta_ohmic = np.zeros_like(t)
    V_cell = np.zeros_like(t)

    for j in range(len(t)):

        Cs = C_surface[j]

        E_eq[j] = params.E0 + (R*params.T)/(params.n*F)*np.log(Cs)

        eta_act[j] = solve_overpotential(i_actual[j], Cs, params)

        eta_ohmic[j] = i_actual[j]*params.L/params.kappa

        V_cell[j] = abs(E_eq[j] + eta_act[j] + eta_ohmic[j])

    return {
        't': t,
        'C': C,
        'C_surface': C_surface,
        'delta_mm': delta_mm,
        'i_actual': i_actual,
        'V_cell': V_cell
    }

results = {}

for iapp in params.i_app_sweep:

    results[iapp] = run_model(iapp)


"========= POST-PROCESSING ========="
# Plotting formatting:
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=12)

ncolors = len(params.i_app_sweep)
ind_colors = np.linspace(0, 1.15, ncolors)
cmap = colormaps['plasma']
colors = cmap(ind_colors)


"========= GLOBAL PLOT FORMATTING ========="

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True
})

line_width = 2.2
legend_x = 1.02


"========= PLOT CONCENTRATION PROFILES ========="

fig, ax = plt.subplots(figsize=(9.5,3.5))

for k, iapp in enumerate(params.i_app_sweep):

    C = results[iapp]['C']

    ax.plot(x*1e3,
            C[:, -1],
            color=colors[k],
            linewidth=line_width,
            label=rf'{iapp:.0f} A m$^{{-2}}$')

ax.set_xlabel('Position [mm]')
ax.set_ylabel(r'Fe$^{2+}$ Concentration [mol m$^{-3}$]')

ax.set_xlim([0, x[-1]*1e3])

ax.legend(frameon=False,
          loc='center left',
          bbox_to_anchor=(legend_x,0.5))

fig.tight_layout(rect=[0,0,0.82,1])

plt.show()


"========= PLOT SURFACE CONCENTRATION ========="

fig2, ax2 = plt.subplots(figsize=(9.5,3.5))

for k, iapp in enumerate(params.i_app_sweep):

    ax2.plot(results[iapp]['t'],
             results[iapp]['C_surface'],
             color=colors[k],
             linewidth=line_width,
             label=rf'{iapp:.0f} A m$^{{-2}}$')

ax2.set_xlabel('Time [s]')
ax2.set_ylabel(r'Surface Concentration [mol m$^{-3}$]')

ax2.set_xlim([0, results[iapp]['t'][-1]])

ax2.legend(frameon=False,
           loc='center left',
           bbox_to_anchor=(legend_x,0.5))

fig2.tight_layout(rect=[0,0,0.82,1])

plt.show()


"========= PLOT DEPOSIT THICKNESS ========="

fig3, ax3 = plt.subplots(figsize=(9.5,3.5))

for k, iapp in enumerate(params.i_app_sweep):

    ax3.plot(results[iapp]['t'],
             results[iapp]['delta_mm'],
             color=colors[k],
             linewidth=line_width,
             label=rf'{iapp:.0f} A m$^{{-2}}$')

ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Deposit Thickness [mm]')

ax3.set_xlim([0, results[iapp]['t'][-1]])

ax3.legend(frameon=False,
           loc='center left',
           bbox_to_anchor=(legend_x,0.5))

fig3.tight_layout(rect=[0,0,0.82,1])

plt.show()


"========= PLOT CURRENT LIMITING ========="

fig4, ax4 = plt.subplots(figsize=(9.5,3.5))

for k, iapp in enumerate(params.i_app_sweep):

    ax4.plot(results[iapp]['t'],
             results[iapp]['i_actual'],
             color=colors[k],
             linewidth=line_width,
             label=rf'{iapp:.0f} A m$^{{-2}}$')

    ax4.axhline(iapp,
                color=colors[k],
                linestyle='--',
                linewidth=1.2,
                alpha=0.8)

ax4.set_xlabel('Time [s]')
ax4.set_ylabel(r'Current Density [A m$^{-2}$]')

ax4.set_xlim([0, results[iapp]['t'][-1]])
ax4.set_ylim([0,2500])

ax4.legend(frameon=False,
           loc='center left',
           bbox_to_anchor=(legend_x,0.5))

fig4.tight_layout(rect=[0,0,0.82,1])

plt.show()


"========= PLOT CELL VOLTAGE ========="

fig5, ax5 = plt.subplots(figsize=(9.5,3.5))

for k, iapp in enumerate(params.i_app_sweep):

    ax5.plot(results[iapp]['t'],
             results[iapp]['V_cell'],
             color=colors[k],
             linewidth=line_width,
             label=rf'{iapp:.0f} A m$^{{-2}}$')

ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Cell Voltage [V]')

ax5.set_xlim([0, results[iapp]['t'][-1]])

ax5.legend(frameon=False,
           loc='center left',
           bbox_to_anchor=(legend_x,0.5))

fig5.tight_layout(rect=[0,0,0.82,1])

plt.show()


"========= PLOT NORMALIZED CURRENT ========="

fig6, ax6 = plt.subplots(figsize=(9.5,3.5))

ax6.axhline(1.0,
            color='black',
            linestyle='--',
            linewidth=1.2,
            label='Ideal')

for k, iapp in enumerate(params.i_app_sweep):

    i_norm = results[iapp]['i_actual']/iapp

    ax6.plot(results[iapp]['t'],
             i_norm,
             color=colors[k],
             linewidth=line_width,
             label=rf'{iapp:.0f} A m$^{{-2}}$')

ax6.set_xlabel('Time [s]')

ax6.set_ylabel(r'Current Utilization, $i/i_{app}$')

ax6.set_xlim([0, results[iapp]['t'][-1]])
ax6.set_ylim([0,1.05])

ax6.legend(frameon=False,
           loc='center left',
           bbox_to_anchor=(legend_x,0.5))

fig6.tight_layout(rect=[0,0,0.82,1])

plt.show()


"========= SAVE FIGURES ========="

fig.savefig('concentration_profiles.png', dpi=600, bbox_inches='tight')
fig2.savefig('surface_concentration.png', dpi=600, bbox_inches='tight')
fig3.savefig('deposit_thickness.png', dpi=600, bbox_inches='tight')
fig4.savefig('current_limiting.png', dpi=600, bbox_inches='tight')
fig5.savefig('cell_voltage.png', dpi=600, bbox_inches='tight')
fig6.savefig('normalized_current.png', dpi=600, bbox_inches='tight')