import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numpy.polynomial import Polynomial
from scipy import integrate, optimize
from scipy.interpolate import PchipInterpolator

@njit
def numerov_homogeneous(coefficient, x0, slope, step):
    """Solve x'' = coefficient*x on a uniform mesh."""
    x = np.zeros(coefficient.size)
    x[0] = x0
    x[1] = x0 + step*slope

    step2 = step**2
    step2_over_12 = step2/12.0
    w_previous = x[0]*(1.0 - step2_over_12*coefficient[0])
    w_current = x[1]*(1.0 - step2_over_12*coefficient[1])
    x_current = x[1]
    coefficient_current = coefficient[1]

    for i in range(2, coefficient.size):
        w_next = 2*w_current - w_previous + step2*coefficient_current*x_current
        coefficient_current = coefficient[i]
        x_current = w_next/(1.0 - step2_over_12*coefficient_current)
        x[i] = x_current
        w_previous, w_current = w_current, w_next
    return x

@njit
def radial_coefficient(energy, ell, radius, U_ks):
    return (ell*(ell + 1)/radius + U_ks)/radius - energy

def compute_radial_orbital(energy, ell, radius, U_ks):
    coefficient = radial_coefficient(energy, ell, radius, U_ks)
    u = numerov_homogeneous(
        coefficient[::-1],
        x0=0.0,
        slope=-1e-10,
        step=radius[0] - radius[1],
    )[::-1]
    norm = integrate.simpson(u**2, x=radius)
    return u/np.sqrt(abs(norm))


def shooting_residual(energy, radius, ell, U_ks):
    u = compute_radial_orbital(energy, ell, radius, U_ks)/radius**ell
    polynomial = Polynomial.fit(radius[:4], u[:4], deg=3)
    return polynomial(0.0)


def find_bound_states(radius, ell, n_states, energy_mesh, U_ks):
    states = []
    residual_left = shooting_residual(energy_mesh[0], radius, ell, U_ks)

    for energy_left, energy_right in zip(energy_mesh[:-1], energy_mesh[1:]):
        residual_right = shooting_residual(energy_right, radius, ell, U_ks)

        if residual_left*residual_right < 0.0:
            energy = optimize.brentq(
                shooting_residual,
                energy_left,
                energy_right,
                args=(radius, ell, U_ks),
                xtol=1e-12,
            )
            states.append((ell, energy))
            if len(states) >= n_states:
                break

        residual_left = residual_right

    return states


def fill_orbitals(states, radius, n_electrons, U_ks, verbose=False):
    rho = np.zeros(radius.size)
    eigenvalue_sum = 0.0
    electron_count = 0
    occupied = []

    for ell, energy in states:
        capacity = 2*(2*ell + 1)
        occupation = min(capacity, n_electrons - electron_count)
        if occupation <= 0:
            break

        u = compute_radial_orbital(energy, ell, radius, U_ks)
        rho += occupation*u**2/(4*np.pi*radius**2)
        eigenvalue_sum += occupation*energy
        electron_count += occupation
        occupied.append((ell, energy, occupation, capacity))

        if verbose:
            print(
                f"l={ell}, E={energy/2:11.6f} Ha, "
                f"occupation={occupation}/{capacity}"
            )

        if electron_count >= n_electrons:
            break

    if electron_count != n_electrons:
        raise RuntimeError(
            f"Only {electron_count} of {n_electrons} electrons could be placed. "
            "Increase n_max or improve the energy mesh."
        )

    return rho, eigenvalue_sum, occupied

def sort_orbitals(states, tie_break=1e-5):
    return sorted(states, key=lambda state: state[1] + tie_break*state[0])


def make_energy_mesh(Z_nuclear, n_points=500, energy_shift=0.1):
    # The small positive shift gives fine sampling near the continuum edge.
    # Positive roots in a finite radial box must not automatically be
    # interpreted as physical atomic bound states.
    energy_scale = 1.2*Z_nuclear**2 + energy_shift
    return (
        -np.geomspace(1e-4, energy_scale, n_points)[::-1]
        + energy_shift
    )
radius_demo = np.linspace(1e-6, 100.0, 2000)
Z_eff = 1.0
n_electrons_demo = 8
n_max_demo = 4
U_model = -2*Z_eff*np.ones(radius_demo.size)
energy_mesh_demo = make_energy_mesh(Z_eff, n_points=400)

model_states = []
for ell in range(n_max_demo):
    model_states += find_bound_states(
        radius_demo, ell, n_max_demo - ell, energy_mesh_demo, U_model
    )
model_states = sort_orbitals(model_states, tie_break=1e-3)
rho_demo, _, occupied_demo = fill_orbitals(
    model_states, radius_demo, n_electrons_demo, U_model, verbose=True
)

normalization = integrate.simpson(
    4*np.pi*radius_demo**2*rho_demo, x=radius_demo
)
print("Integrated density =", normalization)

plt.plot(
    radius_demo,
    4*np.pi*radius_demo**2*rho_demo,
    label="initial radial density",
)
plt.xlim(0.0, 25.0)
plt.xlabel(r"$r/a_0$")
plt.ylabel(r"$4\pi r^2\rho(r)$")
plt.legend()
plt.show()