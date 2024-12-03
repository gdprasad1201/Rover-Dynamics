import matplotlib.pyplot as plt
import time
import numpy as np
from SkidSteeringRobot import SkidSteeringRobot
from RRTStarTorque import RRTStarTorque
from RRTStarManhattan import RRTStarManhattan
from RRTStarCombined import RRTStarCombined

# Gains
gains = {
    'kinematic': [1.0, 1.0],  # k1, k2, k3
    'dynamic': 2.0
}

# Calculate path length
def calculate_path_length(edges):
    total_length = 0
    for parent, child in edges:
        distance = np.linalg.norm(np.array(parent[:2]) - np.array(child[:2]))
        total_length += distance
    return total_length

# Calculate total torque cost
def calculate_total_torque_cost(edges, robot, gains):
    total_cost = 0
    for parent, child in edges:
        eta = np.zeros(2)  # Assume initial zero velocity
        tau = robot.control_law(np.array(parent), np.array(child), eta, eta, np.zeros_like(eta), gains)
        total_cost += np.sum(np.abs(tau))
    return total_cost
# Testing Harness for All Three Planners
def run_testing_harness():
    # Parameters
    start = [0.0, 0.0, 0.0]
    goal = [2.0, 2.0, 0.0]
    area = [-0.5, 2.5, -0.5, 2.5]
    obstacles = [
        (1.0, 1.0, 0.2),
        (1.5, 1.5, 0.2),
        (2.0, 0.5, 0.2)
    ]
    
    # Instantiate SkidSteeringRobot
    skid_steering_robot = SkidSteeringRobot(
        m=1.0, I=0.0036, r=0.0265, c=0.034, x0=-0.02
    )
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(area[0], area[1])
    ax.set_ylim(area[2], area[3])
    ax.grid(True)

    # Plot obstacles
    for ox, oy, radius in obstacles:
        circle = plt.Circle((ox, oy), radius, color='orange', alpha=0.5)
        ax.add_patch(circle)

    # Run RRT* Torque
    start_time_torque = time.time()
    rrt_star_torque = RRTStarTorque(start, goal, obstacles, skid_steering_robot, area, max_iter=1500)
    rrt_star_torque.plan()
    runtime_torque = time.time() - start_time_torque
    rrt_star_torque.visualize(ax, color='blue', label="RRT* Torque")

    # Run RRT* Manhattan
    start_time_manhattan = time.time()
    rrt_star_manhattan = RRTStarManhattan(start, goal, obstacles, area, max_iter=1500)
    rrt_star_manhattan.plan()
    runtime_manhattan = time.time() - start_time_manhattan
    rrt_star_manhattan.visualize(ax, color='yellow', label="RRT* Manhattan")

    # Run RRT* Combined
    start_time_combined = time.time()
    rrt_star_combined = RRTStarCombined(
        start, goal, obstacles, skid_steering_robot, area, alpha=1.0, beta=.2, max_iter=1500
    )
    rrt_star_combined.plan()
    runtime_combined = time.time() - start_time_combined
    rrt_star_combined.visualize(ax, color='red', label="RRT* Combined")

    # Add legend and show plot
    ax.legend()
    plt.show()

    # Metrics
    path_length_torque = calculate_path_length(rrt_star_torque.edges)
    path_length_manhattan = calculate_path_length(rrt_star_manhattan.edges)
    path_length_combined = calculate_path_length(rrt_star_combined.edges)

    energy_torque = calculate_total_torque_cost(rrt_star_torque.edges, skid_steering_robot, gains)
    energy_manhattan = calculate_total_torque_cost(rrt_star_manhattan.edges, skid_steering_robot, gains)
    energy_combined = calculate_total_torque_cost(rrt_star_combined.edges, skid_steering_robot, gains)

    # Display results
    print(f"Path Lengths:")
    print(f"RRT* Torque: {path_length_torque:.4f} units")
    print(f"RRT* Manhattan: {path_length_manhattan:.4f} units")
    print(f"RRT* Combined: {path_length_combined:.4f} units")

    print("\nEnergy Costs:")
    print(f"RRT* Torque: {energy_torque:.4f} units")
    print(f"RRT* Manhattan: {energy_manhattan:.4f} units")
    print(f"RRT* Combined: {energy_combined:.4f} units")

    print("\nRuntimes:")
    print(f"RRT* Torque Runtime: {runtime_torque:.4f} seconds")
    print(f"RRT* Manhattan Runtime: {runtime_manhattan:.4f} seconds")
    print(f"RRT* Combined Runtime: {runtime_combined:.4f} seconds")


# Run the testing harness
run_testing_harness()
