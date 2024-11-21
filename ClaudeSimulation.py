import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

from OccupancyGrid import LayeredOccupancyGrid
from AStar import AStarPlanner, manhattan_distance, manhattan_distance_with_obstacles, AStarNode, distance, create_astar_layer, true_distance_with_obstacles


class PathPlanningSimulation:
    def __init__(self, width=20, height=20, cell_size=1):
        """
        Initialize path planning simulation

        Args:
            width (int): Grid width in cells
            height (int): Grid height in cells
            cell_size (float): Size of each cell in meters
        """
        # Create occupancy grid
        self.grid = LayeredOccupancyGrid(width, height, cell_size)

        # Create layers
        self.grid.create_layer('obstacle', data_type=int, default_value=0)
        if not self.grid.contains_layer('astar'):
            create_astar_layer(self.grid)

        # A* Planner with Manhattan distance heuristic
        trueDistanceHeuristic= lambda grid, node1, node2 : distance(node1, node2)

        self.planner = AStarPlanner(trueDistanceHeuristic)

    def add_random_obstacles(self, obstacle_probability=0.2):
        """
        Add random obstacles to the grid

        Args:
            obstacle_probability (float): Probability of a cell being an obstacle
        """
        for x in range(self.grid.getWidth()):
            for y in range(self.grid.getHeight()):
                if np.random.random() < obstacle_probability:
                    self.grid.set_cell('obstacle', x, y, 1)

    def plan_path(self, start, goal):
        """
        Plan a path using A* algorithm

        Args:
            start (tuple): Start coordinates in meters
            goal (tuple): Goal coordinates in meters

        Returns:
            list: Planned path or None if no path found
        """
        start_time = time.time()
        path = self.planner.plan(
            self.grid,
            start[0], start[1],
            goal[0], goal[1]
        )

        end_time = time.time()

        print("Planning took {} seconds".format(end_time - start_time))

        return path

    def visualize_path(self, path=None):
        """
        Visualize the grid and path

        Args:
            path (list, optional): List of path nodes
        """
        plt.figure(figsize=(10, 8))

        # Create obstacle map
        obstacle_map = np.copy((self.grid.getLayer('obstacle')))

        # Custom colormap
        cmap = ListedColormap(['white', 'gray'])
        plt.imshow(obstacle_map, cmap=cmap, interpolation='nearest')

        # Plot path if exists
        if path:

            print("path created:", path)
            # Extract x and y coordinates
            path_x = [node.x for node in path]
            path_y = [node.y for node in path]

            plt.plot(path_x, path_y, color='red', linewidth=2, marker='o')
            plt.plot(path_x[0], path_y[0], color='green', marker='o', markersize=10, label='Start')
            plt.plot(path_x[-1], path_y[-1], color='blue', marker='o', markersize=10, label='Goal')

        plt.title('A* Path Planning')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, color='lightgray', linestyle='--')
        plt.legend()
        plt.show()


# Demonstration
def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create simulation
    sim = PathPlanningSimulation(width=30, height=30)

    # Add random obstacles
    sim.add_random_obstacles(obstacle_probability=0.25)

    # Define start and goal in meters
    start = (2.5, 2.5)  # 5th cell in both x and y
    goal = (15, 15)  # Around middle of the grid

    # Plan path
    path = sim.plan_path(start, goal)

    # Visualize
    sim.visualize_path(path)


if __name__ == "__main__":
    main()