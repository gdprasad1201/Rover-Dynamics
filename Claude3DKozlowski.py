import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class TerrainAdaptiveSkidSteeringRobot:
    def __init__(self,
                 mass: float = 1.0,  # kg
                 inertia: float = 0.0036,  # kg·m²
                 wheel_radius: float = 0.0265,  # m
                 wheel_track: float = 0.068,  # m (2c)
                 initial_state: np.ndarray = None):
        """
        Initialize the skid-steering mobile robot with terrain adaptation.

        Args:
            mass: Total mass of the robot
            inertia: Moment of inertia
            wheel_radius: Effective radius of wheels
            wheel_track: Distance between left and right wheels
            initial_state: Initial [x, y, theta, vx, vy, omega] state
        """
        # Physical parameters
        self.m = mass
        self.I = inertia
        self.r = wheel_radius
        self.c = wheel_track / 2

        # State variables
        if initial_state is None:
            self.state = np.zeros(6)  # [x, y, theta, vx, vy, omega]
        else:
            self.state = initial_state

        # Friction and terrain parameters
        self.mu_s = 0.1  # Longitudinal friction coefficient
        self.mu_l = 0.5  # Lateral friction coefficient

        # Terrain gradient parameters
        self.terrain_map = None

    def generate_terrain_map(self,
                             width: int = 100,
                             height: int = 100,
                             roughness: float = 0.1) -> np.ndarray:
        """
        Generate a 2D height map representing terrain gradient.

        Args:
            width: Width of terrain map
            height: Height of terrain map
            roughness: Terrain roughness factor

        Returns:
            2D numpy array representing terrain height
        """
        # Generate random terrain using Perlin-like noise
        terrain = np.random.randn(width, height)
        for i in range(1, width):
            for j in range(1, height):
                terrain[i, j] = (terrain[i - 1, j] + terrain[i, j - 1]) / 2 + \
                                roughness * np.random.randn()

        self.terrain_map = terrain
        return terrain

    def calculate_terrain_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Calculate terrain gradient at a given position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Tuple of (x_gradient, y_gradient)
        """
        if self.terrain_map is None:
            return 0, 0

        # Map robot position to terrain map
        map_x = int(x * self.terrain_map.shape[0])
        map_y = int(y * self.terrain_map.shape[1])

        # Ensure within bounds
        map_x = max(0, min(map_x, self.terrain_map.shape[0] - 2))
        map_y = max(0, min(map_y, self.terrain_map.shape[1] - 2))

        # Calculate gradient using central differences
        x_gradient = (self.terrain_map[map_x + 1, map_y] -
                      self.terrain_map[map_x, map_y]) / 1
        y_gradient = (self.terrain_map[map_x, map_y + 1] -
                      self.terrain_map[map_x, map_y]) / 1

        return x_gradient, y_gradient

    def dynamics_model(self,
                       state: np.ndarray,
                       torques: np.ndarray,
                       dt: float) -> np.ndarray:
        """
        Calculate robot dynamics with terrain gradient effects.

        Args:
            state: Current robot state
            torques: Applied wheel torques
            dt: Time step

        Returns:
            Updated robot state
        """
        x, y, theta, vx, vy, omega = state

        # Get terrain gradient at current position
        grad_x, grad_y = self.calculate_terrain_gradient(x, y)

        # Modify torques based on terrain gradient
        terrain_torque_adjustment = np.array([
            grad_x * self.m * 9.81,  # X-direction terrain effect
            grad_y * self.m * 9.81  # Y-direction terrain effect
        ])

        # Wheel angular velocities from torques
        omega_L = (torques[0] + terrain_torque_adjustment[0]) / (self.r * self.m)
        omega_R = (torques[1] + terrain_torque_adjustment[1]) / (self.r * self.m)

        # Calculate longitudinal and angular velocities
        vx_new = self.r * (omega_L + omega_R) / 2
        omega_new = self.r * (omega_R - omega_L) / (2 * self.c)

        # Simple kinematic update with terrain gradient influence
        x_new = x + vx_new * np.cos(theta) * dt
        y_new = y + vx_new * np.sin(theta) * dt
        theta_new = theta + omega_new * dt

        return np.array([x_new, y_new, theta_new, vx_new, vy, omega_new])

    def simulate(self,
                 torque_profile: List[np.ndarray],
                 dt: float = 0.01) -> List[np.ndarray]:
        """
        Simulate robot motion over a series of torque inputs.

        Args:
            torque_profile: List of torque inputs for each time step
            dt: Time step for simulation

        Returns:
            List of robot states over time
        """
        # Generate terrain map
        self.generate_terrain_map()

        states = [self.state]
        for torques in torque_profile:
            self.state = self.dynamics_model(self.state, torques, dt)
            states.append(self.state)

        return states

    def visualize_trajectory(self, states: List[np.ndarray]):
        """
        Visualize robot trajectory and terrain map.

        Args:
            states: List of robot states
        """
        # Extract x and y coordinates
        x_coords = [state[0] for state in states]
        y_coords = [state[1] for state in states]

        plt.figure(figsize=(12, 5))

        # Terrain map subplot
        plt.subplot(121)
        plt.imshow(self.terrain_map, cmap='terrain')
        plt.title('Terrain Height Map')
        plt.colorbar(label='Height')

        # Trajectory subplot
        plt.subplot(122)
        plt.plot(x_coords, y_coords, 'r-', label='Robot Trajectory')
        plt.title('Robot Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


# Example usage
def main():
    # Create robot
    robot = TerrainAdaptiveSkidSteeringRobot()

    # Create a simple torque profile
    torque_profile = [
        np.array([0.1, 0.1]),  # Initial torque
        np.array([0.2, 0.15]),  # Slight increase
        np.array([0.1, 0.2]),  # Adjust torque
        np.array([0.15, 0.15])  # Stabilize
    ]

    # Simulate robot motion
    states = robot.simulate(torque_profile)

    # Visualize results
    robot.visualize_trajectory(states)


if __name__ == "__main__":
    main()