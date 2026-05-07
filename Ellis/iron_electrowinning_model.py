# iron_electrowinning_model.py

"========= IMPORT MODULES ========="
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import font_manager
import numpy as np
from scipy.integrate import solve_ivp

# Plotting formatting:
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=12)
ncolors = 2
ind_colors = np.linspace(0, 1.15, ncolors)
cmap = colormaps['plasma']
colors = cmap(ind_colors)


"========= LOAD INPUTS AND OTHER PARAMETERS ========="

nvars = 2   # Number of state variables

class params:
    # Operating conditions
    i_app = 100        # [A/m^2] Applied current density
    T = 298            # [K] Temperature
    
    # Electrochemistry
    n = 2              # [-] electrons transferred
    i0 = 10            # [A/m^2] exchange current density
    beta = 0.5         # [-] symmetry factor (class convention)
    
    # Thermodynamics
    E0 = -0.44         # [V] standard potential for Fe2+/Fe
    
    # Physical properties
    M = 0.055845       # [kg/mol] molar mass of Fe
    rho = 7870         # [kg/m^3] density of Fe
    
    # System definition
    A = 1.0        # m^2 (industrial scale cathode)
    L = 0.02       # m (elyte thickness)
    V = A * L      # m^3 (elyte volume)


# Positions in solution vector
class ptr:
    C_Fe = 0
    delta = 1


# Constants
R = 8.314      # [J/mol-K]
F = 96485      # [C/mol]


"========= INITIALIZE MODEL ========="
SV_0 = np.zeros((nvars,))
SV_0[ptr.C_Fe] = 1000     # [mol/m^3] initial concentration
SV_0[ptr.delta] = 0.0     # [m] initial thickness


"========= BUTLER VOLMER ========="
def butler_volmer(i0, eta, p):
    return i0 * (np.exp(-p.beta*p.n*F*eta/(R*p.T)) - np.exp((1-p.beta)*p.n*F*eta/(R*p.T)))


"========= DEFINE RESIDUAL FUNCTION ========="
def derivative(t, SV, params, ptr):
    dSV_dt = np.zeros_like(SV)

    # Unpack variables
    C = SV[ptr.C_Fe]
    delta = SV[ptr.delta]

    # Stop if concentration gets too low
    if C <= 1e-6:
        dSV_dt[:] = 0
        return dSV_dt

    # Electrichemistry

    # Nernst equation
    E_eq = params.E0 + (R * params.T) / (params.n * F) * np.log(C)

    # Sign convention:
    # Positive current = reduction (Fe2+ -> Fe)
    eta = -E_eq

    # Butler-Volmer current (currently unused)
    i_BV = butler_volmer(params.i0, eta, params)

    # Galvanostatic condition (current is imposed)
    i = params.i_app

    # Governing Equations

    # Species balance
    dCdt = -params.i_app * params.A / (params.n * F * params.V)

    # Moving boundary
    dDeltadt = (params.M / (params.n * F * params.rho)) * i

    # Store derivatives
    dSV_dt[ptr.C_Fe] = dCdt
    dSV_dt[ptr.delta] = dDeltadt

    return dSV_dt


"========= RUN / INTEGRATE MODEL ========="
solution = solve_ivp(
    derivative,
    [0, 3600],
    SV_0,
    args=(params, ptr),
    t_eval=np.linspace(0, 3600, 200)
)


"========= POST-PROCESSING & PLOTTING ========="
# Extract solution
C = solution.y[ptr.C_Fe]
delta = solution.y[ptr.delta]
t = solution.t

# Convert thickness to mm
delta_mm = delta * 1e3

# Create figure
fig, ax1 = plt.subplots()
fig.set_size_inches((4,3))

# Concentration (left axis)
line1, = ax1.plot(t, C, color='purple', label='Fe$^{2+}$ Concentration')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Concentration [mol/m$^3$]', color='purple')
ax1.tick_params(axis='y', labelcolor='purple')

# Thickness (right axis, now in mm)
ax2 = ax1.twinx()
line2, = ax2.plot(t, delta_mm, color='gold', label='Deposit Thickness')
ax2.set_ylabel('Thickness [mm]', color='gold')
ax2.tick_params(axis='y', labelcolor='gold')

# Combined legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, prop=font, frameon=False)

fig.tight_layout()
plt.show()