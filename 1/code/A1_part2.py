import numpy as np
import matplotlib.pyplot as plt

def line_mse(m, b):

    # Use current slope and intercept to predict the y
    y_pred = m*x_data + b

    # Calculate the MSE
    mse = np.mean((y_pred - y_data)**2)

    return mse

def fit_line_newton(m0=0.0, b0=0.0, tolerance=1e-7, max_iter=500):

    # Start iterating with the chosen initial values (0,0)
    m = m0
    b = b0

    # Record each iteration
    tracker = []

    for i in range(max_iter):

        # Partial derivatives wrt to m
        dMSE_dm = 7*m + 3*b - 15.5
        d2MSE_dm2 = 7

        # Update m, treating b as a constant
        next_m = m - dMSE_dm / d2MSE_dm2

        # Partial derivatives wrt to b
        # Update b using the new m calculated above
        dMSE_db = 3*next_m + 2*b - 6.5
        d2MSE_db2 = 2
        next_b = b - dMSE_db / d2MSE_db2

        # Calculate MSE for this iteration
        mse = line_mse(next_m, next_b)

        # Save iteration
        tracker.append((i + 1, next_m, next_b, mse))

        # Check if convergence is within tolerance
        if (abs(next_m - m) < tolerance and abs(next_b - b) < tolerance):
            m = next_m
            b = next_b

            break

        # Update values for next iteration
        m = next_m
        b = next_b

    final_mse = line_mse(m, b)

    return m, b, final_mse, tracker

def plot_line_iterations(tracker, m0=0.0, b0=0.0):

    # Generates lots of x values to plot the line smoothly
    x_curve = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 300)

    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(
        x_data,
        y_data,
        color="black",
        zorder=5,
        label="Data points"
    )

    # Initial line using intial values
    y_initial = m0*x_curve + b0

    plt.plot(
        x_curve,
        y_initial,
        "--",
        color="gray",
        linewidth=1.5,
        label=f"Initial: m={m0:.2f}, b={b0:.2f}"
    )

    # New colour for each iteration
    colours = plt.cm.turbo(np.linspace(0.1, 0.9, len(tracker)))

    # Only these iterations appear in legend (need to trim otherwise you can't see all the info in the figure)
    legend_iterations = {1, 2, 5, 10, 20, 30, 50, 80, 100}

    # Plots each iteration except the last
    for i, (iteration, m, b, mse) in enumerate(tracker[:-1]):

        # Calculates the line for each iteration
        y_curve = m*x_curve + b

        # Adding to legend
        if iteration in legend_iterations:
            label = (
                f"Iteration {iteration}: "
                f"m={m:.3f}, b={b:.3f}"
            )

        else:
            label = None

        # Plots the link calculated above
        plt.plot(
            x_curve,
            y_curve,
            color=colours[i],
            linewidth=1,
            alpha=0.8,
            label=label
        )

    # Final line parameters 
    final_m = tracker[-1][1]
    final_b = tracker[-1][2]
    final_mse = tracker[-1][3]

    # Plots the (final) curve of best fist
    plt.plot(
        x_curve,
        final_m*x_curve + final_b,
        color="black",
        linewidth=2.5,
        label=(
            f"Final: y={final_m:.3f}x{final_b:+.3f}, "
            f"MSE={final_mse:.4f}"
        )
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Iterative Newton-Raphson Line Fit")

    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()

def parabola_mse(a, b, c):

    # Uses the current coefficients to predict y
    y_pred = (a*x_data**2 + b*x_data + c)

    mse = np.mean((y_pred - y_data)**2)

    return mse

def fit_parabola_newton(a0=0.0, b0=0.0, c0=0.0, tolerance=1e-7, max_iter=500):

    # Chosen initial values
    a = a0
    b = b0
    c = c0

    tracker = []

    for i in range(max_iter):
        # Partial derivatives wrt to a
        # Update a, treating b and c as a constants
        dMSE_da = (49*a + 18*b + 7*c - 41.5)
        d2MSE_da2 = 49
        next_a = a - dMSE_da / d2MSE_da2

        # Partial derivatives wrt to b
        # Update b, using the previous a and treating c as a constant
        dMSE_db = (18*next_a + 7*b + 3*c - 15.5)
        d2MSE_db2 = 7
        next_b = b - dMSE_db / d2MSE_db2

        # Partial derivatives wrt to c
        # Update c, using the previous a and b
        dMSE_dc = (7*next_a + 3*next_b + 2*c - 6.5)
        d2MSE_dc2 = 2
        next_c = c - dMSE_dc / d2MSE_dc2

        # Calculate MSE for this iteration
        mse = parabola_mse(next_a, next_b, next_c)

        # Save iteration
        tracker.append((i + 1, next_a, next_b, next_c, mse))

        # Check convergence
        if (abs(next_a - a) < tolerance and abs(next_b - b) < tolerance and abs(next_c - c) < tolerance):
            a = next_a
            b = next_b
            c = next_c

            break

        # Update values for next iteration
        a = next_a
        b = next_b
        c = next_c

    final_mse = parabola_mse(a, b, c)

    return a, b, c, final_mse, tracker

def plot_parabola_iterations(tracker, a0=0.0, b0=0.0, c0=0.0):
    # Generates lots of x values to plot the line smoothly
    x_curve = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 300)

    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(
        x_data,
        y_data,
        color="black",
        zorder=5,
        label="Data points"
    )

    # Initial parabola from intial coefficients
    y_initial = (a0*x_curve**2 + b0*x_curve + c0)

    plt.plot(
        x_curve,
        y_initial,
        "--",
        color="gray",
        linewidth=1.5,
        label=(
            f"Initial: "
            f"a={a0:.2f}, b={b0:.2f}, c={c0:.2f}"
        )
    )

    # Each iteration has its own colour
    colours = plt.cm.turbo(np.linspace(0.1, 0.9, len(tracker)))

    # Only selected iterations in legend (GS search has many iterations, iteration list must be truncated for proper display)
    legend_iterations = {1, 2, 5, 10, 25, 50, 100}

    for i, (iteration, a, b, c, mse) in enumerate(tracker[:-1]):

        # Calculate parabola for current iteration
        y_curve = (a*x_curve**2 + b*x_curve + c)

        # Add to legend
        if iteration in legend_iterations:
            label = (
                f"Iteration {iteration}: "
                f"a={a:.3f}, "
                f"b={b:.3f}, "
                f"c={c:.3f}"
            )

        else:
            label = None

        # Plot the calculated parabola
        plt.plot(
            x_curve,
            y_curve,
            color=colours[i],
            linewidth=1,
            alpha=0.7,
            label=label
        )

    # Final parabola parameters for parabola of best fit
    final_a = tracker[-1][1]
    final_b = tracker[-1][2]
    final_c = tracker[-1][3]
    final_mse = tracker[-1][4]

    final_curve = (final_a*x_curve**2 + final_b*x_curve + final_c)

    # Plot the final parabola
    plt.plot(
        x_curve,
        final_curve,
        color="black",
        linewidth=2.5,
        label=(
            f"Final: "
            f"{final_a:.3f}x² "
            f"{final_b:+.3f}x "
            f"{final_c:+.3f}, "
            f"MSE={final_mse:.4f}"
        )
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Iterative Newton-Raphson Parabola Fit")

    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()

# Given data points
x_data = np.array([0, 2, 1, 3], dtype=float)
y_data = np.array([0.5, 3.5, 1.5, 7.5], dtype=float)

m, b, line_error, line_history = fit_line_newton(m0=0, b0=0)

plot_line_iterations(line_history, m0=0, b0=0)
print("\nLINE FIT")

print(f"Final m = {m:.8f}")
print(f"Final b = {b:.8f}")
print(f"Final MSE = {line_error:.8f}")

print("\nIterations:")

print(
    f"{'Iteration':<12}"
    f"{'m':<18}"
    f"{'b':<18}"
    f"{'MSE':<18}"
)

for iteration, m_i, b_i, mse_i in line_history:

    print(
        f"{iteration:<12}"
        f"{m_i:<18.8f}"
        f"{b_i:<18.8f}"
        f"{mse_i:<18.8f}"
    )

a, b, c, parabola_error, parabola_history = fit_parabola_newton(
    a0=0,
    b0=0,
    c0=0
)

plot_parabola_iterations(parabola_history, a0=0, b0=0, c0=0)
print("\nPARABOLA FIT")

print(f"Final a = {a:.8f}")
print(f"Final b = {b:.8f}")
print(f"Final c = {c:.8f}")
print(f"Final MSE = {parabola_error:.8f}")

print("\nIterations:")

print(
    f"{'Iteration':<12}"
    f"{'a':<18}"
    f"{'b':<18}"
    f"{'c':<18}"
    f"{'MSE':<18}"
)

for iteration, a_i, b_i, c_i, mse_i in parabola_history:

    print(
        f"{iteration:<12}"
        f"{a_i:<18.8f}"
        f"{b_i:<18.8f}"
        f"{c_i:<18.8f}"
        f"{mse_i:<18.8f}"
    )