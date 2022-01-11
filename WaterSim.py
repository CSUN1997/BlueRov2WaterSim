import numpy as np


class Water:
    def __init__(self):
        self.X_u = -7.2
        self.Y_v = -7.7
        self.N_r = -3
        self.X_u_dot = -2.9
        self.Y_v_dot = -3
        self.N_r_dot = -3.3
        self.L_h = 0.145
        self.W_h = 0.1
        self.gamma_1 = -np.pi / 4
        self.gamma_2 = np.pi / 4
        self.gamma_3 = 3 * np.pi / 4
        self.gamma_4 = -3 * np.pi / 4
        self.V_min = 0
        self.V_max = 0.1
        self.delta_t = 0.1
        self.m = 10
        self.I_z = 2
        self.x_g = 0.01
        self.y_g = 0.01

        self.eta = None
        self.eta_dot = None
        self.eta_dotdot = None
        self.t = None

    def reset_robot(self, x=0.2, y=-0.5, theta=np.pi / 18):
        self.t = 0
        self.eta = [x, y, theta]
        self.eta_dot = np.zeros(3)
        self.eta_dotdot = np.zeros(3)

    def update(self):
        self.eta_dot += self.eta_dotdot
        self.eta += self.eta_dot

    def get_acceleration(self, thrust_force):
        # velocity should be the relative velocity
        x, y, psi = self.eta
        u, v, r = self.eta_dot
        m, x_g, y_g = self.m, self.x_g, self.y_g
        gamma_1, gamma_2, gamma_3, gamma_4 = self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4
        M_eta = np.asarray([[m, 0, -m * (y_g * np.cos(psi) + x_g * np.sin(psi))],
                            [0, m, m * (x_g * np.cos(psi) - y_g * np.sin(psi))],
                            [-m * (y_g * np.cos(psi)), m * (x_g * np.cos(psi) - y_g * np.sin(psi)), self.I_z]])
        C_eta = np.asarray([[0, m * r, -m * np.cos(psi) * (v + r * x_g) - m * np.sin(psi) * (u - r * y_g)],
                            [-m * r, 0, m * np.cos(psi) * (u - r * y_g) - m * np.sin(psi) * (v + r * x_g)],
                            [m * (v * np.cos(psi) + u * np.sin(psi)), -m * (u * np.cos(psi) - v * np.sin(psi)), 0]])
        B11 = np.cos(gamma_1 + psi)
        B12 = np.cos(gamma_2) * np.cos(psi) - np.sin(gamma_1) * np.sin(psi)
        B13 = np.cos(gamma_3) * np.cos(psi) - np.sin(gamma_1) * np.sin(psi)
        B14 = np.cos(gamma_4) * np.cos(psi) - np.sin(gamma_1) * np.sin(psi)
        B21 = np.sin(gamma_1 + psi)
        B22 = np.cos(psi) * np.sin(gamma_1) + np.cos(gamma_2) * np.sin(psi)
        B23 = np.cos(psi) * np.sin(gamma_1) - np.sin(gamma_3) * np.sin(psi)
        B24 = np.cos(psi) * np.sin(gamma_1) + np.cos(gamma_4) * np.sin(psi)
        B31 = self.L_h * np.sin(gamma_1) - self.W_h * np.cos(gamma_1)
        B32 = self.W_h * np.cos(gamma_2) + self.L_h * np.sin(gamma_2)
        B33 = self.W_h * np.cos(gamma_3) - self.L_h * np.sin(gamma_3)
        B34 = -self.W_h * np.cos(gamma_4) - self.L_h * np.sin(gamma_4)
        B_eta = np.asarray([[B11, B12, B13, B14],
                            [B21, B22, B23, B24],
                            [B31, B32, B33, B34]])
        M_inv = np.linalg.inv(M_eta)
        self.eta_dotdot = -M_inv @ C_eta @ self.eta_dot.T + M_inv @ B_eta @ thrust_force.T


if __name__ == '__main__':
    env = Water()
    thrust_force = np.zeros(4)
    env.reset_robot()
    print(env.eta, env.eta_dot, env.eta_dotdot)
    env.get_acceleration(thrust_force)
    env.update()
    print(env.eta, env.eta_dot, env.eta_dotdot)