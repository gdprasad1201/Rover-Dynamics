import numpy as np

from abc import ABC, abstractmethod

# TODO: occupancy grid, check the math, make sure the gradient works

    # RobotState
    # state[0] = x
    # state[1] = y
    # state[2] = z
    # state[3] = theta
    # state[4] = v_x
    # state[5] = v_y
    # state[6] = v_z
    # state[7] = omega

    # WheelTorqueCommand
    # fl: 0
    # fr: 1
    # bl: 2
    # br: 3

class RoverDynamics(ABC):
    state:np.ndarray = np.full(shape=(8,), fill_value=0.0)

    # motor_forces should be [fl, fr, bl, br]
    @abstractmethod
    def apply_update(self, state:np.ndarray, motor_forces:np.ndarray, dt:float, grade:float) -> np.ndarray:
        pass

    def update(self, motor_forces:np.ndarray, dt:float, grade:float):
        self.state = self.apply_update(self.state, motor_forces, dt, grade)

    @abstractmethod
    def apply_mpc(self, state:np.ndarray, target_point:tuple[float, float, float], dt:float, horizon:int=10, slope:float=0.0):
        pass

    def mpc(self, target_point:tuple[float, float, float], dt:float, horizon:int=10, slope:float=0.0):
        return self.apply_mpc(self.state, target_point, dt, horizon, slope)


    def set_state(self, x, y, z, theta, v_x, v_y, v_z, omega):
        self.state = np.array([x, y, z, theta, v_x, v_y, v_z, omega])

    def get_state(self) -> np.ndarray:
        return self.state

