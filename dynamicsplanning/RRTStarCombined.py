import numpy as np
import random
import time
import matplotlib.pyplot as plt

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

class RRTStarCombined:
    def __init__(self, start, goal, obstacles, robot, area, alpha=1.0, beta=1.0, max_iter=500, step_size=0.1, dt=0.01):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.robot = robot
        self.area = area
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.step_size = step_size
        self.dt = dt  # Integration timestep
        self.nodes = [(self.start, 0)]  # (position, cost)
        self.edges = []  # List of edges as (parent, child)

    def is_collision(self, q):
        for ox, oy, radius in self.obstacles:
            if np.linalg.norm(q[:2] - np.array([ox, oy])) < radius:
                return True
        return False

    def sample_point(self):
        x = random.uniform(self.area[0], self.area[1])
        y = random.uniform(self.area[2], self.area[3])
        theta = random.uniform(-np.pi, np.pi)
        return np.array([x, y, theta])

    def nearest_node(self, q):
        return min(self.nodes, key=lambda n: np.sum(np.abs(n[0][:2] - q[:2])))

    def steer(self, q_nearest, q_rand):
        """
        Use robot dynamics to simulate motion from q_nearest toward q_rand.
        """
        q = np.array(q_nearest[:3])
        eta = np.zeros(2)  # Assume zero velocity initially
        max_steps = int(self.step_size / self.dt)

        for _ in range(max_steps):
            tau = self.robot.control_law(q, q_rand, eta, eta, np.zeros(2), gains)
            q_dot = self.robot.kinematic_model
            q_dot = self.robot.kinematic_model(q, eta)
            eta_dot = self.robot.dynamic_model(eta, tau)

            # Update states
            q += q_dot * self.dt
            eta += eta_dot * self.dt

            # Check for collisions
            if self.is_collision(q):
                return None

        return q  # Final state after steering

    def combined_cost(self, q1, q2):
        """
        Calculate the combined cost using Manhattan distance and torque cost.
        """
        manhattan_cost = np.sum(np.abs(q1[:2] - q2[:2]))
        eta = np.zeros(2)  # Assume initial zero velocity
        tau = self.robot.control_law(q1, q2, eta, eta, np.zeros_like(eta), gains)
        torque_cost = np.sum(np.abs(tau))
        return self.alpha * manhattan_cost + self.beta * torque_cost

    def plan(self):
        """
        Plan the path using the RRT* Combined algorithm.
        """
        for _ in range(self.max_iter):
            q_rand = self.sample_point()
            q_nearest = self.nearest_node(q_rand)
            q_new = self.steer(q_nearest[0], q_rand)

            if q_new is None or self.is_collision(q_new):
                continue

            combined_cost = self.combined_cost(q_nearest[0], q_new)
            new_cost = q_nearest[1] + combined_cost
            self.nodes.append((q_new, new_cost))
            self.edges.append((q_nearest[0], q_new))

            if np.linalg.norm(q_new[:2] - self.goal[:2]) < self.step_size:
                self.nodes.append((self.goal, new_cost))
                self.edges.append((q_new, self.goal))
                break

    def visualize(self, ax, color, label):
        """
        Visualize the tree and path.
        """
        for parent, child in self.edges:
            ax.plot([parent[0], child[0]], [parent[1], child[1]], color=color)
        ax.scatter(self.start[0], self.start[1], color='green', label='Start' if label else None)
        ax.scatter(self.goal[0], self.goal[1], color='red', label='Goal' if label else None)
