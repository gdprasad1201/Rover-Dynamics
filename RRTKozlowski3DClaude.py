import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt


class RRTStarRobotPlanner:
    def __init__(
            self,
            robot: 'TerrainAdaptiveSkidSteeringRobot',
            start: np.ndarray,
            goal: np.ndarray,
            bounds: Tuple[np.ndarray, np.ndarray],
            max_iter: int = 1000,
            goal_bias: float = 0.1,
            step_size: float = 0.1,
            goal_threshold: float = 0.2,
            rewire_radius: float = 0.5
    ):
        """
        Initialize RRT* planner for terrain-adaptive robot navigation.

        Args:
            robot: Terrain-adaptive skid steering robot instance
            start: Starting configuration [x, y, theta]
            goal: Goal configuration [x, y, theta]
            bounds: Tuple of (min_bounds, max_bounds) for sampling
            max_iter: Maximum number of iterations
            goal_bias: Probability of sampling goal instead of random point
            step_size: Maximum distance between nodes
            goal_threshold: Distance threshold to consider goal reached
            rewire_radius: Radius for rewiring and finding near nodes
        """
        self.robot = robot
        self.start = start
        self.goal = goal
        self.min_bounds, self.max_bounds = bounds
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.rewire_radius = rewire_radius

        # Tree storage
        self.nodes = [start]
        self.parents = {0: None}
        self.costs = {0: 0.0}

        # Cost function configuration
        self.distance_weight = 1.0
        self.torque_weight = 0.5

    def custom_cost_function(
            self,
            from_node: np.ndarray,
            to_node: np.ndarray,
            torques: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute cost between nodes with multiple components.

        Args:
            from_node: Starting node configuration
            to_node: Target node configuration
            torques: Optional applied torques for dynamics-based cost

        Returns:
            Total weighted cost
        """
        # Euclidean distance cost
        distance_cost = np.linalg.norm(to_node[:2] - from_node[:2])

        # Terrain gradient cost (penalty for steep terrain)
        x_grad, y_grad = self.robot.calculate_terrain_gradient(to_node[0], to_node[1])
        terrain_cost = np.sqrt(x_grad ** 2 + y_grad ** 2)

        # Torque-based cost (if torques are provided)
        torque_cost = 0
        if torques is not None:
            torque_cost = np.linalg.norm(torques)

        # Weighted combination of costs
        total_cost = (
                self.distance_weight * distance_cost +
                self.distance_weight * terrain_cost +
                self.torque_weight * torque_cost
        )

        return total_cost

    def sample_configuration(self) -> np.ndarray:
        """
        Sample a random configuration with goal bias.

        Returns:
            Sampled configuration [x, y, theta]
        """
        if np.random.random() < self.goal_bias:
            return self.goal

        # Random configuration within bounds
        x = np.random.uniform(self.min_bounds[0], self.max_bounds[0])
        y = np.random.uniform(self.min_bounds[1], self.max_bounds[1])
        theta = np.random.uniform(-np.pi, np.pi)

        return np.array([x, y, theta])

    def nearest_node(self, sample: np.ndarray) -> int:
        """
        Find nearest node in the tree to the sample.

        Args:
            sample: Configuration to find nearest neighbor for

        Returns:
            Index of nearest node
        """
        distances = [np.linalg.norm(sample[:2] - node[:2]) for node in self.nodes]
        return np.argmin(distances)

    def steer(self, from_node: np.ndarray, to_node: np.ndarray) -> np.ndarray:
        """
        Generate a node that moves from from_node towards to_node.

        Args:
            from_node: Starting configuration
            to_node: Target configuration

        Returns:
            New configuration
        """
        direction = to_node - from_node
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_node

        scaled_direction = direction * (self.step_size / distance)
        return from_node + scaled_direction

    def is_collision_free(
            self,
            from_node: np.ndarray,
            to_node: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if path between nodes is collision-free.

        Returns:
            Tuple of (is_free, interpolated_torques)
        """
        # Simple interpolation for torques and collision checking
        num_steps = int(np.linalg.norm(to_node[:2] - from_node[:2]) / 0.1)
        torques_sequence = []

        for i in range(num_steps):
            alpha = i / num_steps
            interpolated_node = from_node + alpha * (to_node - from_node)

            # Basic torque generation (could be more sophisticated)
            torques = np.random.uniform(-0.5, 0.5, 2)
            robot.
            torques_sequence.append(torques)

        return True, torques_sequence

    def rrt_star_plan(self) -> List[np.ndarray]:
        """
        Perform RRT* path planning.

        Returns:
            Path from start to goal
        """
        for _ in range(self.max_iter):
            # Sample configuration
            sample = self.sample_configuration()

            # Find nearest node
            nearest_idx = self.nearest_node(sample)
            nearest_node = self.nodes[nearest_idx]

            # Steer towards sample
            new_node = self.steer(nearest_node, sample)

            # Check collision and feasibility
            is_free, torques = self.is_collision_free(nearest_node, new_node)
            if not is_free:
                continue

            # Compute cost to new node
            new_node_cost = (
                    self.costs[nearest_idx] +
                    self.custom_cost_function(nearest_node, new_node, torques[0])
            )

            # Find near nodes for potential connection and rewiring
            near_indices = [
                idx for idx, node in enumerate(self.nodes)
                if np.linalg.norm(node[:2] - new_node[:2]) < self.rewire_radius
            ]

            # Choose best parent
            best_parent_idx = nearest_idx
            best_cost = new_node_cost

            for near_idx in near_indices:
                near_node = self.nodes[near_idx]
                is_connectable, near_torques = self.is_collision_free(near_node, new_node)

                if is_connectable:
                    potential_cost = (
                            self.costs[near_idx] +
                            self.custom_cost_function(near_node, new_node, near_torques[0])
                    )

                    if potential_cost < best_cost:
                        best_parent_idx = near_idx
                        best_cost = potential_cost

            # Add new node
            new_node_idx = len(self.nodes)
            self.nodes.append(new_node)
            self.parents[new_node_idx] = best_parent_idx
            self.costs[new_node_idx] = best_cost

            # Rewire near nodes
            for near_idx in near_indices:
                near_node = self.nodes[near_idx]
                is_connectable, near_torques = self.is_collision_free(new_node, near_node)

                if is_connectable:
                    new_near_cost = (
                            best_cost +
                            self.custom_cost_function(new_node, near_node, near_torques[0])
                    )

                    if new_near_cost < self.costs[near_idx]:
                        self.parents[near_idx] = new_node_idx
                        self.costs[near_idx] = new_near_cost

            # Check goal proximity
            if np.linalg.norm(new_node[:2] - self.goal[:2]) < self.goal_threshold:
                break

        # Reconstruct path
        path = self.reconstruct_path()
        return path

    def reconstruct_path(self) -> List[np.ndarray]:
        """
        Reconstruct the best path from start to goal.

        Returns:
            List of path configurations
        """
        current_idx = len(self.nodes) - 1
        path = [self.nodes[current_idx]]

        while self.parents[current_idx] is not None:
            current_idx = self.parents[current_idx]
            path.append(self.nodes[current_idx])

        return list(reversed(path))

    def plot_path(self, path: List[np.ndarray]):
        """
        Plot the RRT* path and terrain.

        Args:
            path: List of path configurations
        """
        plt.figure(figsize=(12, 5))

        # Terrain map
        plt.subplot(121)
        plt.imshow(self.robot.terrain_map, cmap='terrain')
        plt.title('Terrain Height Map')
        plt.colorbar(label='Height')

        # Path visualization
        plt.subplot(122)

        # Plot all nodes
        node_xs = [node[0] for node in self.nodes]
        node_ys = [node[1] for node in self.nodes]
        plt.scatter(node_xs, node_ys, c='blue', alpha=0.1, label='Nodes')

        # Plot path
        path_xs = [node[0] for node in path]
        path_ys = [node[1] for node in path]
        plt.plot(path_xs, path_ys, 'r-', linewidth=2, label='Optimal Path')

        plt.title('RRT* Path')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Example demonstration
def main():
    from Claude3DKozlowski import TerrainAdaptiveSkidSteeringRobot

    # Create robot
    robot = TerrainAdaptiveSkidSteeringRobot()
    robot.generate_terrain_map()

    # Define problem
    start = np.array([0.1, 0.1, 0])
    goal = np.array([0.9, 0.9, 0])
    bounds = (np.array([0, 0]), np.array([1, 1]))

    # Create RRT* planner
    rrt_planner = RRTStarRobotPlanner(
        robot=robot,
        start=start,
        goal=goal,
        bounds=bounds
    )

    # Plan path
    path = rrt_planner.rrt_star_plan()

    # Visualize
    rrt_planner.plot_path(path)


if __name__ == "__main__":
    main()