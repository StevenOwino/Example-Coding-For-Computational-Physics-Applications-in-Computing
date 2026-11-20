import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Physical parameters
m1, m2 = 1.0, 1.0       # masses (kg)
l1, l2 = 1.0, 1.0       # rod lengths (m)
g = 9.81                 # gravitational acceleration (m/s^2)

# Initial conditions: [theta1, theta2, omega1, omega2]
initial_state = [
    np.radians(120),     # theta1
    np.radians(-10),     # theta2
    0.0,                 # angular velocity 1
    0.0                  # angular velocity 2
]

def derivatives(t, state):
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2

    # Mass matrix M(theta) * [alpha1, alpha2] = right-hand side
    mass_matrix = np.array([
        [(m1 + m2) * l1**2,       m2 * l1 * l2 * np.cos(delta)],
        [m2 * l1 * l2 * np.cos(delta), m2 * l2**2]
    ])

    rhs = np.array([
        -m2 * l1 * l2 * np.sin(delta) * omega2**2
        - (m1 + m2) * g * l1 * np.sin(theta1),

         m2 * l1 * l2 * np.sin(delta) * omega1**2
        - m2 * g * l2 * np.sin(theta2)
    ])

    alpha1, alpha2 = np.linalg.solve(mass_matrix, rhs)
    return [omega1, omega2, alpha1, alpha2]

# Numerically integrate the equations
duration = 20.0
fps = 30
frame_count = int(duration * fps)
times = np.linspace(0, duration, frame_count)

solution = solve_ivp(
    derivatives,
    (0, duration),
    initial_state,
    t_eval=times,
    rtol=1e-9,
    atol=1e-9
)

# Convert angles to bob coordinates
theta1, theta2 = solution.y[0], solution.y[1]
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# Set up the animation
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-(l1 + l2 + 0.2), l1 + l2 + 0.2)
ax.set_ylim(-(l1 + l2 + 0.2), l1 + l2 + 0.2)
ax.grid(True)
ax.set_title("Double Pendulum")

rod, = ax.plot([], [], "o-", lw=2, markersize=8)
trace, = ax.plot([], [], "-", lw=1, alpha=0.6)
trace_x, trace_y = [], []

def update(frame):
    rod.set_data([0, x1[frame], x2[frame]],
                 [0, y1[frame], y2[frame]])

    trace_x.append(x2[frame])
    trace_y.append(y2[frame])
    trace.set_data(trace_x, trace_y)

    return rod, trace

animation = FuncAnimation(
    fig,
    update,
    frames=frame_count,
    interval=1000 / fps,
    blit=True
)

plt.show()
