import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Gains
gains = {
    'kinematic': [1.0, 1.0, 1.0],  # k1, k2, k3
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


class RRTStarManhattan:
    def __init__(self, start, goal, obstacles, area, max_iter=500, step_size=0.1):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.area = area
        self.max_iter = max_iter
        self.step_size = step_size
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
        direction = q_rand[:2] - q_nearest[:2]
        direction /= np.linalg.norm(direction)
        q_new = q_nearest[:2] + direction * self.step_size
        theta_new = np.arctan2(direction[1], direction[0])
        return np.array([q_new[0], q_new[1], theta_new])
    
    def plan(self):
        for _ in range(self.max_iter):
            q_rand = self.sample_point()
            q_nearest = self.nearest_node(q_rand)
            q_new = self.steer(q_nearest[0], q_rand)
            
            if self.is_collision(q_new):
                continue
            
            manhattan_cost = np.sum(np.abs(q_nearest[0][:2] - q_new[:2]))
            new_cost = q_nearest[1] + manhattan_cost
            self.nodes.append((q_new, new_cost))
            self.edges.append((q_nearest[0], q_new))
            
            if np.linalg.norm(q_new[:2] - self.goal[:2]) < self.step_size:
                self.nodes.append((self.goal, new_cost))
                self.edges.append((q_new, self.goal))
                break
    
    def visualize(self, ax, color, label):
        for parent, child in self.edges:
            ax.plot([parent[0], child[0]], [parent[1], child[1]], color=color)
        ax.scatter(self.start[0], self.start[1], color='green', label='Start' if label else None)
        ax.scatter(self.goal[0], self.goal[1], color='red', label='Goal' if label else None)
