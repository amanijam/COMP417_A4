import numpy as np

class KalmanFilter1():
    def __init__(self, dt, l, m, M, g):
        """
        :param dt: sampling time (time for 1 cycle)
        :param l: half length of pendulum
        :param m: mass of pendulum
        :param M: mass of cart
        :param g: gravity
        """
        # Define sampling time
        self.dt = dt

        # Initial State [theta, theta_dot]
        # x: 2x1
        self.x = np.matrix([[0], [0]])

        inertia_p = m * ((2*l)**2)
        c = (m+M)*inertia_p + M*m*(l**2)
        
        # Define the State Transition Matrix F
        # F: 2x2 
        # function to estimate theta and theta dot
        self.F = np.matrix([[0, 1],
                            [((M+m)*m*l*g)/c, 0]]) # assume 0 friction

        # Define the Control Input Matrix B
        # B: 2x1
        self.B = np.matrix([[0], [(m*l)/c]])

        # Define Measurement Mapping Matrix
        # H is a 1x2 matrix
        self.H = np.matrix([[1, 0]])

        # Initial Process Noise Covariance
        # Q: 2x2 matrix
        q_c = (self.dt) # coefficient of Q
        self.Q = q_c * np.eye(2)

        # Initial Measurement Noise Covariance
        # R: 1x1
        self.R = 0.01

        # Initial Covariance Matrix P (must be symmetric and positive semi-definite)
        # P: 2x2 
        p_c = 0.01 # coefficient of P
        self.P = p_c * np.eye(2)

    # Must be called BEFORE update
    # self.x becomes x_(t+1|t)
    # self.P becomes P_(t+1|t)
    # return x_(t+1|t)
    def propagate(self, u):
        """
        :param u: control input, acceleration we incur on the system
        """
        # State estimate is updated from system dynamics
        # x_(t+1|t) = F*x_(t|t) + B*u_(t) 
        # F: 2x2
        # x: 2x1
        # F*x => 2x1
        # B: 2x1
        # u: 1x1
        # B*x => 2x1
        # result: 2x1
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)

        # Uncertainty estimate GROWS
        # P_(t+1|t) = F*P_(t|t)*F' + Q      (G?????)   
        # G is not necessary. However, ignoring G means you did not make any difference
        #  between the uncertainty in the states. The bigger the covarience matrix, the bigger the uncertainty
        # F: 2x2, F': 2x2
        # P: 2x2
        # F*P*F' => 2x2
        # Q: 2x2
        # result: F*P*F' + Q => 2x2
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        #return self.x

    def update(self, z):
        """
        :param z: observation, theta
        """
        # z_(t+1) = theta = H * x_(t+1|t)
        # H is a vector that, when multiplied by x, returns a SCALAR BC Z IS THE ANGLE
        # H: 1x2
        # x: 2x1
        # result: H*x => 1x1
        # Expected value of sensor reading
        z_hat = self.H*self.x

        # Difference between expect and true observation (theta)
        r = z - z_hat

        # Covariance of sensor reading
        # S = H*P_(t+1|t)*H' + R
        # S: 1x1
        # H: 1x2
        # P: 2x2
        # H*P: 1x2
        # H': 2x1
        # (H*P)*H': 1x1
        # R: 1x1
        # result: H*P*H' + R => 1x1
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman Gain (how much to correct est.)
        # K = P_(t+1|t)*H'*inv(S)
        # K: 2x1
        # P: 2x2
        # H: 1x2, H': 2x1
        # P*H' => 2x1
        # S: 1x1
        # result: P*H'*S => 2x1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 

        # Correct state estimate by adding residual times gains
        # x_(t+1|t+1) = x_(t+1|t) + K*r
        # x: 2x1
        # K: 2x1
        # r: 1x1
        # result: x+K*r => 2x1
        self.x = self.x + np.dot(K, r)

        # Uncertainty estimate SHRINKS
        # P_(t+1|t+1) = P_(t+1|t) - P_(t+1|t)*H'*inv(S)*H*P_(t+1|t)
        # P: 2x2
        # H: 1x2, H': 2x1
        # S: 1x1
        # result: P + P*H'*inv(S)*H*P => 2x2
        self.P = self.P - np.dot(np.dot(np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)), self.H), self.P)

        return np.squeeze(np.asarray(self.x))