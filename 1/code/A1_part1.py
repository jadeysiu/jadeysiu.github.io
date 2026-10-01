import numpy as np
import matplotlib.pyplot as plt
import math

def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100):
    # Start with an chosen x value
    x = initial_guess

    # Records the estimate and distance from every iteration 
    tracker = []

    for i in range(max_iter):

        # First derivative of squared distance
        D_prime = 2*(x - x0) + 2*(f(x) - y0)*df(x)

        # Second derivative of squared distance
        D_double_prime = (2 + 2*(df(x)**2) + 2*(f(x) - y0)*ddf(x))

        # Newton-Raphson update
        next_x = x - D_prime / D_double_prime

        # Calculate distance for current iteration
        distance = math.sqrt((next_x - x0)**2 + (f(next_x) - y0)**2)

        # Save current iteration info
        tracker.append((i+1, next_x, distance))

        # Stop when tolerance is reached
        if abs(next_x - x) < tolerance:
            x = next_x
            break

        x = next_x

    shortest_distance = math.sqrt((x - x0)**2 + (f(x) - y0)**2)

    return shortest_distance, x, tracker

def plot_newton(x0, y0, f, tracker, minimum_distance):

    # Final iteration x and y coordinates
    optimum_x = tracker[-1][1]
    optimum_y = f(optimum_x)

    # Get x-values from each iteration
    iteration_x = [item[1] for item in tracker]

    # For displaying purposes: decides how much of the x-axis to show
    x_values = [x0] + iteration_x
    x_min = min(x_values) - 2
    x_max = max(x_values) + 2

    # Generates a lot of x-values for the parabola so it's displayed as a smooth curve
    x_curve = np.linspace(x_min, x_max, 1000)
    y_curve = [f(x) for x in x_curve]

    plt.figure(figsize=(10, 6))

    ax = plt.gca()

    # Put grid behind everything
    ax.set_axisbelow(True)
    plt.grid(zorder=0, alpha=0.4)

    # Plots the function
    plt.plot(
        x_curve,
        y_curve,
        color="black",
        linewidth=2,
        zorder=1,
        label="f(x)"
    )

    # Plots the given point
    plt.scatter(
        x0,
        y0,
        color="red",
        # s=55,
        zorder=5,
        label=f"Given point ({x0}, {y0})"
    )

    # Colours for iteration points
    colours = plt.cm.turbo(np.linspace(0.1, 0.9, len(tracker)))

    # Plot each iteration except the last
    for i, (iteration, x, distance) in enumerate(tracker[:-1]):

        plt.scatter(
            x,
            f(x),
            color=colours[i],
            # edgecolor="black",
            linewidth=0.5,
            # s=sizes[i],
            zorder=5,
            label=f"Iteration {iteration}: x = {x:.4f}"
        )

    final_iteration, optimum_x, minimum_distance = tracker[-1]
    optimum_y = f(optimum_x)

    plt.scatter(
            optimum_x,
            optimum_y,
            color="black",
            edgecolors="white",
            linewidth=0.5,
            s=60,
            marker="*",
            zorder=6,
            label=f"Minimum point: ({optimum_x:.4f},{optimum_y:.4f})"
        )

    # Dotted minimum-distance line
    plt.plot(
        [x0, optimum_x],
        [y0, optimum_y],
        "k--",
        linewidth=1.5,
        zorder=2,
        label=f"Minimum distance = {minimum_distance:.4f}"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Newton-Raphson: Point ({x0}, {y0})")

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):

    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi

    # Squared distance from given point to curve
    def dist_sq(x):
        return (x - x0)**2 + (f(x) - y0)**2

    # Creates the 2 search points inside the interval
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
     
    # Store the estimate for each iteration
    tracker = []
    iteration = 1

    while abs(b - a) > tolerance:

        if f_x1 < f_x2:

            b = x2
            x2 = x1
            f_x2 = f_x1

            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)

        else:

            a = x1
            x1 = x2
            f_x1 = f_x2

            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)

        # Use midpoint as the estimate of minimum
        current_x = (a + b) / 2

        # Current distance
        distance = math.sqrt(dist_sq(current_x))

        # Save iteration info
        tracker.append((iteration, current_x, distance))

        iteration += 1

    best_x = (a + b) / 2

    shortest_distance = math.sqrt(dist_sq(best_x))

    return shortest_distance, best_x, tracker

def plot_golden(x0, y0, f, tracker, minimum_distance):

    # Final iteration x and y coordinates
    optimum_x = tracker[-1][1]
    optimum_y = f(optimum_x)

    # Get x-values for each iteration
    iteration_x = [item[1] for item in tracker]

    # For displaying purposes: decides how much of the x-axis to show
    x_values = [x0] + iteration_x
    x_min = min(x_values) - 2
    x_max = max(x_values) + 2

    # Generate a bunch of x values for the parabola
    x_curve = np.linspace(x_min, x_max, 1000)
    y_curve = [f(x) for x in x_curve]

    # Wider figure to leave room for legend
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Grid behind everything
    ax.set_axisbelow(True)
    plt.grid(zorder=0, alpha=0.4)

    # Plots the function
    plt.plot(
        x_curve,
        y_curve,
        color="black",
        linewidth=2,
        zorder=1,
        label="f(x)"
    )

    # Plot the given point
    plt.scatter(
        x0,
        y0,
        color="red",
        zorder=5,
        label=f"Given point ({x0}, {y0})"
    )

    # Colours for all Golden Section iterations
    colours = plt.cm.turbo(np.linspace(0.1, 0.9, len(tracker)))

    # Only these iterations appear in legend (need to trim otherwise you can't see all the info in the figure)
    legend_iterations = {1, 5, 10, 20, 30, 50, 80, 100}

    # Plot every iteration except final one
    for i, (iteration, x, distance) in enumerate(tracker[:-1]):

        if iteration in legend_iterations:
            point_label = f"Iteration {iteration}: x = {x:.4f}"
        else:
            point_label = None

        plt.scatter(
            x,
            f(x),
            color=colours[i],
            linewidth=0.5,
            zorder=5,
            label=point_label
        )

    # Final iteration becomes minimum star
    final_iteration, optimum_x, minimum_distance = tracker[-1]
    optimum_y = f(optimum_x)

    plt.scatter(
        optimum_x,
        optimum_y,
        color="gold",
        edgecolor="black",
        linewidth=0.5,
        s=60,
        marker="*",
        zorder=6,
        label=(
            f"Minimum point: "
            f"({optimum_x:.4f},{optimum_y:.4f})"
        )
    )

    # Minimum-distance line
    plt.plot(
        [x0, optimum_x],
        [y0, optimum_y],
        "k--",
        linewidth=1.5,
        zorder=2,
        label=f"Minimum distance = {minimum_distance:.4f}"
    )

    # Labels
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Golden Section Search: Point ({x0}, {y0})"
    )

    # Smaller legend - all iterations are plotted but the legend is truncated
    plt.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=9
    )

    # Reserve space for legend
    plt.subplots_adjust(right=0.72)

    plt.show()

# Function
f = lambda x: x**2 + 5
df = lambda x: 2*x
ddf = lambda x: 2

# Points to test
points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]

for x0, y0 in points:

    distance, best_x, tracker = find_distance_newton(x0, y0, f, df, ddf, initial_guess=0)

    print("\n--------------------------------")
    print("Newton-Raphson")
    print("Given point:", (x0, y0))

    print("\nIterations:")
    print(f"{'Iteration':<12}{'x value':<18}{'Distance':<18}")

    for iteration, x, dist in tracker:

        print(
            f"{iteration:<12}"
            f"{x:<18.8f}"
            f"{dist:<18.8f}"
        )

    print("\nFinal Results:")
    print(f"Minimum x: {best_x:.8f}")
    print(f"Minimum point: ({best_x:.8f}, {f(best_x):.8f})")
    print(f"Minimum distance: {distance:.8f}")
    print(f"Number of iterations: {len(tracker) - 1}")

    plot_newton(x0, y0, f, tracker, distance)

    distance, best_x, tracker = golden_section_search(x0, y0, f, a=-2, b=2)

    print("\n--------------------------------")
    print("Golden Section Search")
    print("Given point:", (x0, y0))

    print("\nIterations:")
    print(
        f"{'Iteration':<12}"
        f"{'x value':<18}"
        f"{'Distance':<18}"
    )

    for iteration, x, dist in tracker:

        print(
            f"{iteration:<12}"
            f"{x:<18.8f}"
            f"{dist:<18.8f}"
        )

    print("\nFinal Results:")
    print(f"Minimum x: {best_x:.8f}")
    print(
        f"Minimum point: "
        f"({best_x:.8f}, {f(best_x):.8f})"
    )
    print(f"Minimum distance: {distance:.8f}")
    print(f"Number of iterations: {len(tracker)}")

    plot_golden(x0, y0, f, tracker, distance)