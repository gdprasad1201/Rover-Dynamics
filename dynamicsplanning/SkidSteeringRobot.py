import numpy as np

class SkidSteeringRobot:
    def __init__(self, m, I, r, c, x0):
        # Parameters
        self.m = m      # Mass of the robot
        self.I = I      # Inertia
        self.r = r      # Effective radius of wheels
        self.c = c      # Half of the wheel track width
        self.x0 = x0    # Artificial constraint parameter

        # Inertia Matrix
        self.M = np.diag([m, m, I])
    
    def kinematic_model(self, q, eta):
        """Kinematic model: q_dot = S(q) * eta"""
        theta = q[2]
        vx, omega = eta
        S = np.array([
            [np.cos(theta), -self.x0 * np.sin(theta)],
            [np.sin(theta), self.x0 * np.cos(theta)],
            [0, 1]
        ])
        q_dot = S @ eta
        return q_dot

    def dynamic_model(self, eta6, tau):
        """Dynamic model: M_bar * eta_dot + C_bar * eta + R_bar = B_bar * tau"""
        M_bar = np.array([
            [self.m, 0],
            [0, self.I]
        ])
        B_bar = np.array([
            [1 / self.r, 1 / self.r],
            [-self.c / self.r, self.c / self.r]
        ])
        # Assuming simplified dynamics (no friction or Coriolis terms)
        R_bar = np.zeros(2)
        eta_dot_calc = np.linalg.inv(M_bar) @ (B_bar @ tau - R_bar)
        return eta_dot_calc

    def control_law(self, q, qr, eta, eta_ref, eta_dot_ref, gains):
        """Control law combining kinematic and dynamic levels."""
        # Errors
        q_error = q - qr
        eta_error = eta - eta_ref

        # Kinematic Control (simplified example)
        k1, k2 = gains['kinematic']
        eta_ref_dot = -np.array([k1, k2]) * q_error[:2]

        # Dynamic Control (backstepping)
        k3 = gains['dynamic']
        tau = k3 * eta_error + eta_ref_dot
        return tau