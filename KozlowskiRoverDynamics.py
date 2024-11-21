import numpy as np
from RoverDynamics import RoverDynamics
from RickyRoverDynamics import RickyRoverDynamics
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class RobotParams:
    """Physical parameters of the robot"""
    a: float  # Distance from center to front wheels
    b: float  # Distance from center to rear wheels
    c: float  # Half of the wheel track (half distance between left and right wheels)
    r: float  # Effective wheel radius
    m: float  # Mass of the robot
    I: float  # Moment of inertia
    x_icr: float  # x-coordinate of instantaneous center of rotation


@dataclass
class ControlParams:
    """Control parameters"""
    k1: float  # Position gain
    k2: float  # Orientation gain
    k3: float  # Dynamic control gain
    epsilon1: float  # Steady-state error bound
    epsilon2: float  # Robust control parameter
    alpha0: float  # Initial oscillator amplitude
    alpha1: float  # Convergence rate


class KozlowskiRoverDynamics(RoverDynamics):
    def __init__(self, robot_params: RobotParams, control_params: ControlParams):
        self.robot = robot_params
        self.control = control_params

        # Initialize oscillator state
        self.zd = np.array([
            self.control.alpha0 * np.cos(-np.pi / 3),
            self.control.alpha0 * np.sin(-np.pi / 3)
        ])

        # Initialize time
        self.last_time = None

    def compute_rotation_matrix(self, theta: float) -> np.ndarray:
        """Compute the 2D rotation matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    def compute_error_transform(self, theta: float, theta_error: float) -> np.ndarray:
        """Compute the error transformation matrix P"""
        P = np.zeros((3, 3))
        P[0, 0] = -theta_error * np.cos(theta) + 2 * np.sin(theta)
        P[0, 1] = -theta_error * np.sin(theta) - 2 * np.cos(theta)
        P[0, 2] = -2 * self.robot.x_icr
        P[1, 2] = 1
        P[2, 0] = np.cos(theta)
        P[2, 1] = np.sin(theta)
        return P

    def apply_update(self, state: np.ndarray, motor_forces: np.ndarray, dt: float,
                     grade: float) -> np.ndarray:
        pass

    def update_oscillator(self, dt: float):
        """Update the oscillator state"""
        # Compute desired error bound
        delta_d = self.control.alpha0 * np.exp(-self.control.alpha1 * dt) + self.control.epsilon1

        # Update oscillator (simple harmonic motion)
        omega = 2.0  # oscillator frequency
        self.zd[0] = delta_d * np.cos(omega * dt)
        self.zd[1] = delta_d * np.sin(omega * dt)

    # motor_forces should be [fl, fr, bl, br]
    # this isn't really mpc it just computes wheel torques for movement on a planar surface with perfect friction
    def apply_mpc(self, state: np.ndarray, target_point: tuple[float, float, float], dt: float,
                  horizon: int = 10, slope: float = 0.0) -> np.ndarray:
        """
        Compute control inputs for the skid-steering robot

        Args:
            state: [x, y, z, theta, v_x, v_y, v_z, omega] current robot state
            target_point: [x_d, y_d, z_d] desired robot pose
            dt: timestep
            horizon: number of steps to calculate MPC for
            slope: slope

        Returns:
            motor_torques as an np.ndarray [fl, fr, bl, br]
        """

        # Compute errors
        # this error just ignores theta error because there is no target theta only target points
        # it also ignores z because the model doesn't work with that
        error = np.array([state[0] - target_point[0],
                          state[1] - target_point[1],
                          state[3]])

        # Update oscillator
        self.update_oscillator(dt)

        # Compute transformation matrices
        theta = state[2]
        P = self.compute_error_transform(theta, error[2])

        # Transform errors to auxiliary system
        Z = P @ error

        # Compute control in auxiliary coordinates
        u = np.zeros(2)
        u[0] = -self.control.k1 * Z[2]  # Position control
        u[1] = -self.control.k2 * Z[1]  # Orientation control

        # Add oscillator term
        u += self.zd

        # Compute velocity transformation matrix
        L = error[0] * np.sin(theta) - error[1] * np.cos(theta)
        T = np.array([[L, 1], [1, 0]])

        # Transform to robot velocities
        eta = T @ u

        # Convert to wheel velocities using equation (3) from the paper
        left_wheel_velocity = (eta[0] - self.robot.c * eta[1]) / self.robot.r
        right_wheel_velocity = (eta[0] + self.robot.c * eta[1]) / self.robot.r

        return np.array(
            [left_wheel_velocity, right_wheel_velocity, left_wheel_velocity, right_wheel_velocity])


# Example usage
def create_example_controller():
    """Create an example controller with typical parameters"""
    robot_params = RobotParams(
        a=0.039,  # meters
        b=0.039,  # meters
        c=0.034,  # meters
        r=0.0265,  # meters
        m=1.0,  # kg
        I=0.0036,  # kg*m^2
        x_icr=-0.02  # meters
    )

    control_params = ControlParams(
        k1=0.5,
        k2=0.5,
        k3=10.0,
        epsilon1=0.01,
        epsilon2=0.1,
        alpha0=1.5,
        alpha1=0.4
    )

    return KozlowskiRoverDynamics(robot_params, control_params)


# Demo code
if __name__ == "__main__":
    roverController = create_example_controller()
    rover = RickyRoverDynamics(mass=40.0, wheel_base=0.5, track_width=0.5, max_motor_force=50.0)


    rover.set_state(x=0.0, y=0.0, z=0.0, theta=0.0, v_x=0.0, v_y=0.0, v_z=0.0, omega=0.0)

    target = (5.0, 5.0, 0)

    # not fully implemented
    gradient = np.radians(10)

    dt = 0.1
    for t in range(20):
        motor_forces = roverController.apply_mpc(rover.get_state(), target, dt, slope=gradient)
        rover.update(motor_forces, dt, grade=gradient)
        print(f"Time: {t * dt:.1f}s, State: {rover.get_state()}, Motor Forces: {motor_forces}")
