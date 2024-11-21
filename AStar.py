from operator import contains
from webbrowser import open_new

from typing import Union, Dict, Optional, List, Callable

import math

import numpy as np

from sortedcontainers import SortedList

from OccupancyGrid import LayeredOccupancyGrid


class AStarNode:
    x: int
    y: int

    f_val: float = 0
    g_val: float = 0
    h_val: float = 0
    parent = None  # will hold parent node

    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y


def distance(a: AStarNode, b: AStarNode):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def manhattan_distance(node1, node2):
    # Example heuristic: Manhattan distance
    x1, y1 = node1.x, node1.y
    x2, y2 = node2.x, node2.y
    return abs(x1 - x2) + abs(y1 - y2)


def points_near_line(start, end, occupancy_grid:LayeredOccupancyGrid, max_distance = 1):
    # Convert start and end to numpy arrays
    start = np.array(start)
    end = np.array(end)

    # Calculate the direction vector of the line
    direction = end - start

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Calculate the perpendicular vector
    perpendicular = np.array([-direction[1], direction[0]])

    # Get grid dimensions
    height, width = occupancy_grid.getHeight(), occupancy_grid.getWidth()

    # List to store nearby points
    nearby_points = []

    # Iterate through all grid points
    for y in np.arange(0, height, occupancy_grid.getCellSize()):
        for x in np.arange(0, width, occupancy_grid.getCellSize()):
            point = np.array([x, y])

            # Calculate the vector from start to the current point
            to_point = point - start

            # Project this vector onto the line direction
            projection = np.dot(to_point, direction)

            # Check if the projection is within the line segment
            if 0 <= projection <= np.linalg.norm(end - start):
                # Calculate the perpendicular distance
                distance = abs(np.dot(to_point, perpendicular))

                # If the distance is within the threshold, add to nearby points
                if distance <= max_distance:
                    nearby_points.append((x, y))

    return nearby_points


def manhattan_distance_with_obstacles(grid: LayeredOccupancyGrid, node1: AStarNode, node2: AStarNode):
    cost = manhattan_distance(node1, node2)
    for point in points_near_line((node1.x, node1.y), (node2.x, node2.y), grid):
        if grid.get_cell("obstacle", point[0], point[1]):
            cost += 10000
            break
    return cost

def true_distance_with_obstacles(grid: LayeredOccupancyGrid, node1: AStarNode, node2: AStarNode):
    cost = distance(node1, node2)
    for point in points_near_line((node1.x, node1.y), (node2.x, node2.y), grid):
        if grid.get_cell("obstacle", point[0], point[1]):
            cost += 10000
            break
    return cost

def create_astar_layer(grid:LayeredOccupancyGrid):
    grid.create_layer("astar", AStarNode, None)
    for row in range(0, grid.getHeight()):
        for col in range(0, grid.getWidth()):
            grid.set_cell("astar", row, col, AStarNode(row, col))

class AStarPlanner:
    open_nodes = SortedList(key=lambda node: node.f_val)
    closed_nodes = np.ndarray(shape=(0,), dtype=AStarNode)

    obstacle_penalty = 1000
    obstacle_check_distance = 1

    def __init__(self, heuristic: Callable[[LayeredOccupancyGrid, AStarNode, AStarNode], float]):
        self.heuristic = heuristic

    def plan(self, grid: LayeredOccupancyGrid,
             start_x_m: float, start_y_m: float,
             goal_x_m: float, goal_y_m: float
             ) -> list[AStarNode] | None:

        start_x, start_y = grid.get_indices(start_x_m, start_y_m)
        goal_x, goal_y = grid.get_indices(goal_x_m, goal_y_m)

        print("start indices:", start_x, start_y)
        print("goal indices:", goal_x, goal_y)

        max_x = grid.getWidth()
        max_y = grid.getHeight()

        goal_node = grid.get_cell("astar", goal_x, goal_y)
        goal_node.x = goal_x
        goal_node.y = goal_y

        start_node = grid.get_cell("astar", start_x, start_y)
        start_node.x = start_x
        start_node.y = start_y

        # print("start node:", start_node.x, start_node.y)
        # print("goal node:", goal_node.x, goal_node.y)


        start_node.g_val = 0
        start_node.f_val = self.heuristic(grid, start_node, goal_node)
        start_node.f_val = distance(start_node, goal_node)
        start_node.h_val = start_node.f_val

        print("distance to goal:", distance(goal_node, start_node))
        # print("start node has data f:", grid.get_cell("astar", start_x, start_y).f_val)
        # print("start node has data y:", grid.get_cell("astar", start_x, start_y).y)

        self.open_nodes.add(start_node)

        while len(self.open_nodes) > 0:
            curr_node = self.open_nodes.pop(0)

            if curr_node == goal_node:
                path = [curr_node]
                while path[-1] != start_node:
                    path.append(path[-1].parent)
                path.reverse()
                print("found goal node:", curr_node.parent)
                return path

            self.closed_nodes = np.append(self.closed_nodes, curr_node)

            for i in range(-1, 2):
                neighbor_x = curr_node.x + i
                if 0 < neighbor_x < max_x:

                    for j in range(-1, 2):
                        if j == 0 and i == 0:
                            continue
                        neighbor_y = curr_node.y + j
                        if 0 < neighbor_y < max_y:

                            neighbor = grid.get_cell("astar", neighbor_x, neighbor_y)
                            obstacle = grid.get_cell("obstacle", neighbor_x, neighbor_y)
                            neighbor.x = neighbor_x
                            neighbor.y = neighbor_y

                            possible_g_val = neighbor.g_val + self.heuristic(grid, curr_node, neighbor) + (obstacle * self.obstacle_penalty)

                            if neighbor not in self.closed_nodes or possible_g_val < neighbor.g_val:
                                neighbor.g_val = possible_g_val
                                neighbor.h_val = self.heuristic(grid, neighbor, goal_node)
                                neighbor.f_val = neighbor.g_val + neighbor.h_val

                                neighbor.parent = curr_node

                                # if not contains(self.open_nodes, neighbor):
                                self.open_nodes.add(neighbor)

        return None
