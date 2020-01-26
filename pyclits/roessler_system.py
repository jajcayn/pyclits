import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp


def right_side(y, t, params = [0,0,0,0,0,0,0,0]):
    x1, x2, x3, y1, y2, y3 = y      # unpack current values of y
    a1, a2, b1, b2, c1, c2, omega1, omega2, epsilon = params  # unpack parameters
    derivs = [-omega1 * x2 - x3, omega1 * x1 + a1 * x2, b1 + x3 * (x1 - c1), - omega2 * y2 - y3 + epsilon * (x1 - y1), omega2 * y1 + a2 * y2, b2 + y3 * (y1 - c2)]

    return derivs

def right_side_ivp(t, y, params):
    x1, x2, x3, y1, y2, y3 = y      # unpack current values of y
    a1, a2, b1, b2, c1, c2, omega1, omega2, epsilon = params  # unpack parameters
    derivs = [-omega1 * x2 - x3, omega1 * x1 + a1 * x2, b1 + x3 * (x1 - c1), - omega2 * y2 - y3 + epsilon * (x1 - y1), omega2 * y1 + a2 * y2, b2 + y3 * (y1 - c2)]

    return derivs

def roessler_oscillator(**kwargs):
    # Parameters
    if "a1" in kwargs:
        a1 = kwargs["a1"]
    else:
        a1 = 0.15

    if "a2" in kwargs:
        a2 = kwargs["a2"]
    else:
        a2 = 0.15

    if "b1" in kwargs:
        b1 = kwargs["b1"]
    else:
        b1 = 0.2

    if "b2" in kwargs:
        b2 = kwargs["b2"]
    else:
        b2 = 0.2

    if "c1" in kwargs:
        c1 = kwargs["c1"]
    else:
        c1 = 10.0

    if "c2" in kwargs:
        c2 = kwargs["c2"]
    else:
        c2 = 10.0

    if "omega1" in kwargs:
        omega1 = kwargs["omega1"]
    else:
        omega1 = 1.015

    if "omega2" in kwargs:
        omega2 = kwargs["omega2"]
    else:
        omega2 = 0.985

    if "epsilon" in kwargs:
        epsilon = kwargs["epsilon"]
    else:
        epsilon = 0.001

    # Initial values
    if "x1" in kwargs:
        x1 = kwargs["x1"]
    else:
        x1 = 0.0

    if "x2" in kwargs:
        x2 = kwargs["x2"]
    else:
        x2 = 0.0

    if "x3" in kwargs:
        x3 = kwargs["x3"]
    else:
        x3 = 0.0

    if "y1" in kwargs:
        y1 = kwargs["y1"]
    else:
        y1 = 0.0

    if "y2" in kwargs:
        y2 = kwargs["y2"]
    else:
        y2 = 0.0

    if "y3" in kwargs:
        y3 = kwargs["y3"]
    else:
        y3 = 1.0

    if "method" in kwargs:
        method = kwargs["method"]
    else:
        method = "LSODA"

    # Bundle parameters for ODE solver
    params = [a1, a2, b1, b2, c1, c2, omega1, omega2, epsilon]

    # Bundle initial conditions for ODE solver
    y0 = [x1, x2, x3, y1, y2, y3]

    # Make time array for solution
    if "tStart" in kwargs:
        tStart = kwargs["tStart"]
    else:
        tStart = 0.0

    if "tStop" in kwargs:
        tStop = kwargs["tStop"]
    else:
        tStop = 20000.

    if "tInc" in kwargs:
        tInc = kwargs["tInc"]
    else:
        tInc = 0.001

    timepoints = np.arange(tStart, tStop, tInc)

    # Call the ODE solver
    # old version
    #psoln = odeint(right_side, y0, t, args=(params,))

    return solve_ivp(lambda t, y: right_side_ivp(t, y, params), [tStart, tStop], y0, method=method)


def visualization(sol, **kwargs):
    if hasattr(kwargs, "tStart"):
        tStart = kwargs["tStart"]
    else:
        tStart = 0.0

    if hasattr(kwargs, "tStop"):
        tStop = kwargs["tStop"]
    else:
        tStop = 20000.

    fig, axes = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(8, 8))
    plt.tight_layout()
    plt.xlim(tStart, tStop)

    for num, ax in enumerate(axes):
        ax.plot(sol.t, sol.y[num, :])

    plt.xlabel("Time")
    plt.ylabel("Coordinate")

    plt.show()


if __name__ == "__main__":
    sol = roessler_oscillator()

    print(sol)
    # print(sol.y)
    print(sol.y.shape)

    visualization(sol)
