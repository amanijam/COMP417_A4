import numpy as np
from numpy.linalg import inv

class KalmanFilter():
    def __init__(self, dt, l, m_p, m_c, g):
        """
        :param dt: sampling time (time for 1 cycle)
        :param l: half length of pendulum
        :param m_p: mass of pendulum
        :param m_c: mass of cart
        :param g: gravity
        """
        self.dt = dt

        # Initial State [theta, theta_dot]
        # x: 2x1
        self.x_prev = np.array([[0], [0]])
        self.z_last = 0 # last theta observation
        
        # State Transition Matrix F
        # F: 2x2 
        self.F = np.array([ [1, dt],
                            [0, 1]])

        # Control Input Matrix B
        # B: 2x1
        self.B = np.array([ [0],
                            [dt]])

        # Measurement Mapping Matrix
        # H is a 1x2 matrix
        self.H = np.array([[1, 0]])

        # Process Noise Covariance
        # Q: 2x2 matrix
        q_coeff = 0.01**3
        self.Q = q_coeff * np.eye(2)

        # Measurement Noise Covariance
        # R: 1x1
        self.R = 0.01

        # Initial Covariance Matrix P (must be symmetric and positive semi-definite)
        # P: 2x2 
        p_c = 0.01 
        self.P_prev = p_c * np.eye(2)

    def get_P(self):
        return self.P_prev

    def propagate_update(self, u, z):
        """
        :param u: control input, acceleration we incur on the system
        :param z: observation, theta
        """
        ## PREDICT
        u = np.array(u)
        
        # State estimate is updated from system dynamics
        # x_(t|t-1) = F*x_(t-1|t-1) + B*u_(t-1) 
        x = np.matmul(self.F, self.x_prev) + self.B * u

        # Uncertainty estimate GROWS
        # P_(t|t-1) = F*P_(t-1|t-1)*F' + Q       
        # G is not necessary. However, ignoring G means you did not make any difference
        #  between the uncertainty in the states. The bigger the covarience matrix, the bigger the uncertainty
        P = np.matmul(np.matmul(self.F, self.P_prev), np.transpose(self.F)) + self.Q


        ## UPDATE
        z = np.array(z)

        # y_t = z_t - H*x_(t|t-1)
        y_hat = z - np.matmul(self.H, x)

        # Covariance of sensor reading
        # S = H*P_(t|t-1)*H' + R
        S = np.matmul(np.matmul(self.H, P), np.transpose(self.H)) + self.R

        # Kalman Gain (how much to correct est.)
        # K = P_(t|t-1)*H'*inv(S)
        K = np.matmul(np.matmul(P, np.transpose(self.H)), inv(S))

        # Correct state estimate by adding residual times gains
        # x_(t|t) = x_(t|t-1) + K*y_t
        x = x + np.matmul(K, y_hat)

        # Uncertainty estimate SHRINKS
        # P_(t|t) = P_(t|t-1) - P_(t|t-1)*H'*inv(S)*H*P_(t|t-1)
        P = P - np.matmul(np.matmul(np.matmul(np.matmul(P, np.transpose(self.H)), inv(S)), self.H), P)

        self.z_last = z
        self.x_prev = x
        self.P_prev = P

        return (float(x[0][0]), float(x[1][0]))