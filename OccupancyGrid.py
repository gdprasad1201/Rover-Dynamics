import numpy as np
from typing import Union, Dict, Optional, List, Callable


class LayeredOccupancyGrid:
    def __init__(self, width: int, height: int):
        """
        Initialize a layered 2D occupancy grid.

        Args:
            width (int): Total width of the grid
            height (int): Total height of the grid
        """
        self._width = width
        self._height = height
        self._layers: Dict[str, np.ndarray] = {}
        self._cell_widths: Dict[str, float] = {}
        self._layer_update_methods: Dict[str, Callable[[Union[int, float], Union[int, float]], Union[int, float]]] = {}
        self._layer_retrieve_methods: Dict[str, Callable[[Union[int, float]], Union[int, float]]] = {}

    def create_layer(
            self,
            name: str,
            data_type: type = int,
            default_value: Union[int, float] = 0,
            cell_width_m: float = 1.0,
            update_method: Optional[Callable[[Union[int, float], Union[int, float]], Union[int, float]]] = lambda x, y : y,
            retrieve_method: Optional[Callable[[Union[int, float]], Union[int, float]]] = lambda x : x
    ):
        """
        Create a new layer in the grid.

        Args:
            name (str): Unique name for the layer
            data_type (type): Data type of the layer (int or float)
            default_value (Union[int, float]): Default value for cells
            cell_width_m (float): Width of each cell in the layer
            update_method (Optional[Callable]): Function to be applied to each update when it is
                                                applied to the cell. It should receive two items
                                                the current value and the update value and return
                                                the new value
        """
        if name in self._layers:
            raise ValueError(f"Layer '{name}' already exists")

        layer = np.full((self._height, self._width), default_value, dtype=data_type)
        self._layers[name] = layer
        self._cell_widths[name] = cell_width_m
        self._layer_update_methods[name] = update_method
        self._layer_retrieve_methods[name] = retrieve_method

    def set_cell(
            self,
            layer: Union[str, int],
            x: int,
            y: int,
            value: Union[int, float]
    ):
        """
        Set the value of a specific cell in a layer.

        Args:
            layer (Union[str, int]): Layer name or index
            x (int): X coordinate
            y (int): Y coordinate
            value (Union[int, float]): Value to set
        """
        layer_name = self._get_layer_name(layer)

        if x < 0 or x >= self._width or y < 0 or y >= self._height:
            raise IndexError("Cell coordinates out of grid bounds")

        self._layers[layer_name][y, x] = self._layer_update_methods[layer](self._layers[layer_name][y, x], value)

    def get_cell(
            self,
            layer: Union[str, int],
            x: int,
            y: int
    ) -> Union[int, float]:
        """
        Get the value of a specific cell in a layer.

        Args:
            layer (Union[str, int]): Layer name or index
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            Union[int, float]: Cell value
        """
        layer_name = self._get_layer_name(layer)

        if x < 0 or x >= self._width or y < 0 or y >= self._height:
            raise IndexError("Cell coordinates out of grid bounds")

        return self._layer_retrieve_methods[layer_name](self._layers[layer_name][y, x])

    def set_cell_location(self, layer: Union[str, int],
                          x_m: int, y_m: int,
                          value: Union[int, float]
                          ):
        """
        Set the value of a specific cell in a layer.

        Args:
            layer (Union[str, int]): Layer name or index
            x_m (int): X location
            y_m (int): Y location
            value (Union[int, float]): Value to set
        """
        layer_name = self._get_layer_name(layer)

        # perform ceiling division using double negation
        x_index = -(-x_m // self._cell_widths[layer])
        y_index = -(-y_m // self._cell_widths[layer])

        if x_index < 0 or x_index >= self._width or y_index < 0 or y_index >= self._height:
            raise IndexError("Cell position out of grid bounds")

        self._layers[layer_name][y_index, x_index] = self._layer_update_methods[layer](self._layers[layer_name][y_index, x_index], value)

    def get_cell_location(
            self,
            layer: Union[str, int],
            x_m: int,
            y_m: int
    ) -> Union[int, float]:
        """
        Get the value of a specific cell in a layer.

        Args:
            layer (Union[str, int]): Layer name or index
            x_m (int): X location
            y_m (int): Y location

        Returns:
            Union[int, float]: Cell value
        """
        layer_name = self._get_layer_name(layer)

        # perform ceiling division using double negation
        x_index = -(-x_m // self._cell_widths[layer])
        y_index = -(-y_m // self._cell_widths[layer])

        if x_index < 0 or x_index >= self._width or y_index < 0 or y_index >= self._height:
            raise IndexError("Cell position out of grid bounds")

        return self._layer_retrieve_methods[layer_name](self._layers[layer_name][y_index, x_index])

    def _get_layer_name(self, layer: Union[str, int]) -> str:
        """
        Convert layer index or name to layer name.

        Args:
            layer (Union[str, int]): Layer identifier

        Returns:
            str: Layer name
        """
        if isinstance(layer, int):
            layer_names = list(self._layers.keys())
            if 0 <= layer < len(layer_names):
                return layer_names[layer]
            raise IndexError("Layer index out of range")

        if layer not in self._layers:
            raise KeyError(f"Layer '{layer}' does not exist")

        return layer

    def get_layer_info(self, layer: Union[str, int]) -> Dict:
        """
        Get information about a specific layer.

        Args:
            layer (Union[str, int]): Layer identifier

        Returns:
            Dict: Layer information including name, data type, cell width
        """
        layer_name = self._get_layer_name(layer)
        return {
            'name': layer_name,
            'data_type': self._layers[layer_name].dtype,
            'cell_width': self._cell_widths[layer_name],
            'shape': self._layers[layer_name].shape
        }

    def list_layers(self) -> List[str]:
        """
        List all layer names in the grid.

        Returns:
            List[str]: Names of all layers
        """
        return list(self._layers.keys())


def sigmoid(a):
    test_val = 709.78

    # for a_val in a:
    #     if abs(a_val) > test_val:
    #         print("Value will be clipped in sigmoid:", a_val)

    a = np.clip(a, -test_val, test_val)

    return 1 / (1 + np.exp(-a))

# Example usage
if __name__ == "__main__":
    # Create a grid
    grid = LayeredOccupancyGrid(width=10, height=10)

    detection_confidence = 0.68

    step_size = 0.75
    step_size = np.log(detection_confidence / (1-detection_confidence))
    print("Step size:", step_size)
    odds_bounds = 6*0.75

    # Create different layers
    grid.create_layer('elevation', data_type=float, default_value=0.0, cell_width_m=0.5)
    # implements an obstacle layer that uses logOdds
    grid.create_layer('obstacle', data_type=float, default_value=0.2,
                      update_method= lambda x, y : max(1e-7, min(x + step_size*(2*y - 1), 1.0 - 1e-7)),
                      retrieve_method= lambda x : int(np.log(x / (1 - x)) > 0.9)
                      )

    # implements an obstacle layer that uses sigmoid odds
    grid.create_layer('obstacle2', data_type=float, default_value=0,
                      update_method= lambda x, y : max(-odds_bounds, min(x + step_size*(2*y - 1), odds_bounds)),
                      retrieve_method= lambda x : int(sigmoid(x) > 0.9)
                      )
    grid.create_layer('terrain_type', data_type=int, default_value=0)

    # Set some cell values
    grid.set_cell('elevation', 5, 5, 10.25)

    grid.set_cell('obstacle2', 5, 5, 1)
    grid.set_cell('obstacle2', 5, 5, 1)
    # grid.set_cell('obstacle2', 5, 5, 1)
    grid.set_cell('obstacle2', 5, 5, 1)

    print(grid._layers['obstacle2'])
    maxNum = 10
    for i in range(1, maxNum):
        print(i, "sig:", sigmoid(i))

        # i = i / maxNum
        # print(i, ":", np.log(i / (1 - i)))


    grid.set_cell('terrain_type', 5, 5, 2)

    # Retrieve layer information
    print(grid.get_layer_info('obstacle2'))

    # Get cell values
    print(grid.get_cell('elevation', 5, 5))  # Should print 10.25
    print(grid.get_cell('obstacle2', 5, 5))  # Should print 10.25
    print(grid.get_cell('terrain_type', 5, 5))  # Should print 2
