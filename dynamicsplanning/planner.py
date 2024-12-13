from os import stat_result

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


def aggregate_stats(stat_result):
    aggregated_stats = []

    for stat_set in stat_result:
        # torque_stats = np.mean([run[0] for run in stat_set], axis=0)
        manhattan_stats = np.mean([run[1] for run in stat_set], axis=0)
        combined_stats = np.mean([run[2] for run in stat_set], axis=0)

        aggregated_stats.append((
            # torque_stats,
            manhattan_stats, combined_stats))

    return aggregated_stats

# Testing Harness for All Three Planners
def run_testing_harness(_obstacles = None, _verbose = False):
    # Parameters
    start = [0.0, 0.0, 0.0]
    goal = [2.0, 2.0, 0.0]
    area = [-0.5, 2.5, -0.5, 2.5]
    obstacles = _obstacles

    if _obstacles is None:
        obstacles = [
            (1.0, 1.0, 0.2),
            (1.5, 1.5, 0.2),
            (2.0, 0.5, 0.2)
        ]

    
    # Instantiate SkidSteeringRobot
    skid_steering_robot = SkidSteeringRobot(
        m=40.0, I=0.0036, r=0.0265, c=0.034, x0=-0.02
    )
    
    # Plot setup
    if _verbose:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(area[0], area[1])
        ax.set_ylim(area[2], area[3])
        ax.grid(True)

        # Plot obstacles
        for ox, oy, radius in obstacles:
            circle = plt.Circle((ox, oy), radius, color='orange', alpha=0.5)
            ax.add_patch(circle)

    # Run RRT* Torque
    # start_time_torque = time.time()
    # rrt_star_torque = RRTStarTorque(start, goal, obstacles, skid_steering_robot, area, max_iter=1500)
    # rrt_star_torque.plan()
    # runtime_torque = time.time() - start_time_torque
    # rrt_star_torque.visualize(ax, color='blue', label="RRT* Torque")

    # Run RRT* Manhattan
    start_time_manhattan = time.time()
    rrt_star_manhattan = RRTStarManhattan(start, goal, obstacles, area, max_iter=1500)
    rrt_star_manhattan.plan()
    runtime_manhattan = time.time() - start_time_manhattan

    if _verbose:
        rrt_star_manhattan.visualize(ax, color='yellow', label="RRT* Manhattan")

    # Run RRT* Combined
    start_time_combined = time.time()
    rrt_star_combined = RRTStarCombined(
        start, goal, obstacles, skid_steering_robot, area, alpha=1.0, beta=0.5, max_iter=1500
    )
    rrt_star_combined.plan()
    runtime_combined = time.time() - start_time_combined

    if _verbose:
        rrt_star_combined.visualize(ax, color='red', label="RRT* Combined")

    # Add legend and show plot
    if _verbose:
        ax.legend()
        plt.show()

    # Metrics
    # path_length_torque = calculate_path_length(rrt_star_torque.edges)
    path_length_manhattan = calculate_path_length(rrt_star_manhattan.edges)
    path_length_combined = calculate_path_length(rrt_star_combined.edges)

    # energy_torque = calculate_total_torque_cost(rrt_star_torque.edges, skid_steering_robot, gains)
    energy_manhattan = calculate_total_torque_cost(rrt_star_manhattan.edges, skid_steering_robot, gains)
    energy_combined = calculate_total_torque_cost(rrt_star_combined.edges, skid_steering_robot, gains)

    # Display results
    print(f"Path Lengths:")
    # print(f"RRT* Torque: {path_length_torque:.4f} units")
    print(f"RRT* Manhattan: {path_length_manhattan:.4f} units")
    print(f"RRT* Combined: {path_length_combined:.4f} units")

    print("\nEnergy Costs:")
    # print(f"RRT* Torque: {energy_torque:.4f} units")
    print(f"RRT* Manhattan: {energy_manhattan:.4f} units")
    print(f"RRT* Combined: {energy_combined:.4f} units")

    print("\nRuntimes:")
    # print(f"RRT* Torque Runtime: {runtime_torque:.4f} seconds")
    print(f"RRT* Manhattan Runtime: {runtime_manhattan:.4f} seconds")
    print(f"RRT* Combined Runtime: {runtime_combined:.4f} seconds")

    # RRTStarTorqueStats = [path_length_torque, energy_torque, runtime_torque]
    RRTStarManhattanStats = [path_length_manhattan, energy_manhattan, runtime_combined]
    RRTStarCombinedStats = [path_length_combined, energy_combined, runtime_combined]

    return [RRTStarManhattanStats, RRTStarCombinedStats]




if __name__ == "__main__":

    obstacles0 = [
        (1.0, 1.0, 0.2),
        (1.5, 1.5, 0.2),
        (2.0, 0.5, 0.2)
    ]
    # Run the testing harness
    # run_testing_harness(obstacles0)

    obstacles1 = [[1.07, 1.93, 0.09], [0.35, 1.25, 0.24], [0.29, 0.5, 0.24], [0.25, 1.71, 0.26], [1.63, 1.64, 0.06]]
    obstacles2 = [[1.7,1.78,0.24],[0.97,1.78,0.17],[0.39,2.27,0.15],[0.07,2.07,0.11],[1.49,1.25,0.23]]
    obstacles3 = [[1.86,1.00,0.27],[2.19,2.3,0.09],[0.43,0.05,0.23],[1.38,2.15,0.18]]
    obstacles4 = [[0.14,1.17,0.14],[2.20,0.95,0.16]]
    obstacles5 = [[0.47,2.15,0.2],[0.72,1.18,0.08],[1.44,0.5,0.19]]
    obstacles6 = [[0.65,0.15,0.1],[0.23,2.25,0.13],[1.08,0.84,0.17],[0.84,2.18,0.12],[1.94,1.53,0.27]]

    obstacle_list = [obstacles1, obstacles2, obstacles3, obstacles4, obstacles5, obstacles6]
    num_runs = 1

    stat_result = []

    for obstacle_set in obstacle_list:
        print(obstacle_set)
        statSet = []
        for i in range(num_runs):
            statSet.append(run_testing_harness(obstacle_set))

        stat_result.append(statSet)

    # plot the stats for each obstacle set

    # run_testing_harness(obstacles1, True)
    # run_testing_harness(obstacles2, True)
    # run_testing_harness(obstacles3, True)
    # run_testing_harness(obstacles4, True)
    # run_testing_harness(obstacles5, True)
    # run_testing_harness(obstacles6, True)


    if True:
        # Aggregating statistics
        aggregated_stats = aggregate_stats(stat_result)

        # Extract metrics for plotting
        path_lengths = [[stats[0][0], stats[1][0], stats[2][0]] for stats in aggregated_stats]
        energy_costs = [[stats[0][1], stats[1][1], stats[2][1]] for stats in aggregated_stats]
        runtimes = [[stats[0][2], stats[1][2], stats[2][2]] for stats in aggregated_stats]

        labels = [f"Obstacle Set {i + 1}" for i in range(len(aggregated_stats))]

        # Plot Path Lengths
        plt.figure(figsize=(12, 6))
        x = np.arange(len(labels))
        width = 0.25

        # plt.bar(x - width, [p[0] for p in path_lengths], width, label="RRT* Torque", color='blue')
        plt.bar(x, [p[1] for p in path_lengths], width, label="RRT* Manhattan", color='yellow')
        plt.bar(x + width, [p[2] for p in path_lengths], width, label="RRT* Combined", color='red')

        plt.xticks(x, labels)
        plt.ylabel("Path Length (units)")
        plt.title("Average Path Lengths by Obstacle Set and Planner")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Energy Costs
        plt.figure(figsize=(12, 6))

        # plt.bar(x - width, [e[0] for e in energy_costs], width, label="RRT* Torque", color='blue')
        plt.bar(x, [e[1] for e in energy_costs], width, label="RRT* Manhattan", color='yellow')
        plt.bar(x + width, [e[2] for e in energy_costs], width, label="RRT* Combined", color='red')

        plt.xticks(x, labels)
        plt.ylabel("Energy Cost (units)")
        plt.title("Average Energy Costs by Obstacle Set and Planner")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Runtimes
        plt.figure(figsize=(12, 6))

        # plt.bar(x - width, [r[0] for r in runtimes], width, label="RRT* Torque", color='blue')
        plt.bar(x, [r[1] for r in runtimes], width, label="RRT* Manhattan", color='yellow')
        plt.bar(x + width, [r[2] for r in runtimes], width, label="RRT* Combined", color='red')

        plt.xticks(x, labels)
        plt.ylabel("Runtime (seconds)")
        plt.title("Average Runtimes by Obstacle Set and Planner")
        plt.legend()
        plt.grid(True)
        plt.show()






