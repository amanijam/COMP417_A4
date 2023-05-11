import numpy as np


class PID_controller:
    def __init__(self, dt):
        self.prev_error = 0
        self.prev_integral = 0
        self.dt = dt

    def get_action(self, theta, theta_dot, random_controller=False):
        if random_controller:
            return np.random.uniform(-1, 1)
        else:
            # Gains
            Kp = 1.05
            Kd = 0.08
            Ki = 0.008

            error = theta  
            derivative = (error - self.prev_error)/self.dt
            integral = self.prev_integral + error * self.dt

            action = (Kp * error) + (Kd * derivative) + (Ki * integral)

            self.prev_integral = integral
            return action
