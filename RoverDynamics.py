import numpy as np
from scipy.optimize import minimize

# TODO: occupancy grid, check the math, make sure the gradient works

class RoverDynamics:
    def __init__(self, mass = 40.0, wheel_base=0.5, track_width=0.5, max_motor_force=50.0, gravity=9.81):
        self.mass = mass
        self.wheel_base = wheel_base
        self.track_width = track_width
        self.max_motor_force = max_motor_force
        self.gravity = gravity  
        self.inertia = 1.0/12.0 * self.mass * (self.wheel_base**2 + self.track_width**2)
        self.state = np.zeros(8)

    def set_state(self, x, y, z, theta, v_x, v_y, v_z, omega):
        self.state = np.array([x, y, z, theta, v_x, v_y, v_z, omega])

    def get_state(self):
        return self.state
    
    def update(self, motor_forces, dt, grade=0.0):
        motor_forces = np.clip(motor_forces, -self.max_motor_force, self.max_motor_force)

        F_drive = (motor_forces[0] + motor_forces[1] + motor_forces[2] + motor_forces[3]) / 4.0
        F_gravity_along_slope = self.mass * self.gravity * np.sin(grade)
        F_normal = self.mass * self.gravity * np.cos(grade)
        F_x = F_drive * np.cos(grade) - F_gravity_along_slope  
        F_z = F_drive * np.sin(grade)  

        if grade == 0.0:
            F_z = 0.0

        F_y = 0.0

        torque_z = (motor_forces[1] - motor_forces[0]) * self.track_width / 2.0 + (motor_forces[3] - motor_forces[2]) * self.track_width / 2.0

        a_x = F_x / self.mass
        a_y = F_y / self.mass
        a_z = F_z / self.mass

        alpha = torque_z / self.inertia

        v_x = self.state[4] + a_x * dt
        v_y = self.state[5] + a_y * dt
        v_z = self.state[6] + a_z * dt
        omega = self.state[7] + alpha * dt

        x = self.state[0] + (v_x * np.cos(self.state[3]) - v_y * np.sin(self.state[3])) * dt
        y = self.state[1] + (v_x * np.sin(self.state[3]) + v_y * np.cos(self.state[3])) * dt

        z = self.state[2] + v_z * dt
        z = max(0, z)

        theta = self.state[3] + omega * dt

        self.state = np.array([x, y, z, theta, v_x, v_y, v_z, omega])

    
    def mpc(self, target_point, dt, horizon=10, slope=0.0):
        def cost_function(motor_forces):
            temp_state = self.state.copy()
        
            total_cost = 0.0
            for i in range(horizon):
                motor_forces_clipped = np.clip(motor_forces[i*4:(i+1)*4], -self.max_motor_force, self.max_motor_force)
                
                F_drive = (motor_forces_clipped[0] + motor_forces_clipped[1] + motor_forces_clipped[2] + motor_forces_clipped[3]) / 4.0
                F_gravity_along_slope = self.mass * self.gravity * np.sin(slope)
                F_normal = self.mass * self.gravity * np.cos(slope)
                F_x = F_drive * np.cos(slope) - F_gravity_along_slope
                F_z = F_drive * np.sin(slope)

                if slope == 0:
                    F_z = 0.0

                F_y = 0.0
                torque_z = (motor_forces_clipped[1] - motor_forces_clipped[0]) * self.track_width / 2.0 + (motor_forces_clipped[3] - motor_forces_clipped[2]) * self.track_width / 2.0
                
                a_x = F_x / self.mass
                a_y = F_y / self.mass
                a_z = F_z / self.mass
                
                alpha = torque_z / self.inertia
                
                v_x = temp_state[4] + a_x * dt
                v_y = temp_state[5] + a_y * dt
                v_z = temp_state[6] + a_z * dt
                omega = temp_state[7] + alpha * dt
                
                
                x = temp_state[0] + (v_x * np.cos(temp_state[3]) - v_y * np.sin(temp_state[3])) * dt
                y = temp_state[1] + (v_x * np.sin(temp_state[3]) + v_y * np.cos(temp_state[3])) * dt

                z = temp_state[2] + v_z * dt
                z = max(0, z)

                yaw = temp_state[3] + omega * dt
                
                temp_state = np.array([x, y, z, yaw, v_x, v_y, v_z, omega])
                
                distance_error = np.sqrt((x - target_point[0])**2 + (y - target_point[1])**2 + (z - target_point[2])**2)
                total_cost += distance_error
            
            return total_cost
        
        intial_guess = np.zeros(horizon*4)
        bounds = [(-self.max_motor_force, self.max_motor_force) for _ in range(horizon*4)]
        result = minimize(cost_function, intial_guess, bounds=bounds, method='SLSQP')

        optimal_motor_forces = result.x[:4]
        return optimal_motor_forces
    

if __name__ == "__main__":
    rover = RoverDynamics(mass=40.0, wheel_base=0.5, track_width=0.5, max_motor_force=50.0)

    rover.set_state(x=0.0, y=0.0, z=0.0, theta=0.0, v_x=0.0, v_y=0.0, v_z=0.0, omega=0.0)

    target = (5.0, 5.0, 0)

    # not fully implemented
    gradient = np.radians(10)

    dt = 0.1
    for t in range(80):
        motor_forces = rover.mpc(target, dt, slope=gradient)
        rover.update(motor_forces, dt, grade=gradient)
        print(f"Time: {t * dt:.1f}s, State: {rover.get_state()}, Motor Forces: {motor_forces}")

    
        
        