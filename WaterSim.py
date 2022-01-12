import numpy as np
import copy
import matplotlib.pyplot as plt


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

        self.traject = None
        self.vel_traject = None

        self.V_c = None
        self.v_hc = None

    def reset_robot(self, x=0.2, y=-0.5, theta=np.pi / 18):
        self.t = 0
        self.V_c = (self.V_max + self.V_min) / 2
        self.v_hc = [np.zeros(3)]
        self.eta = [x, y, theta]
        self.eta_dot = np.zeros(3)
        self.eta_dotdot = np.zeros(3)
        self.traject = [copy.deepcopy(self.eta), copy.deepcopy(self.eta)]
        self.vel_traject = [copy.deepcopy(self.eta_dot), copy.deepcopy(self.eta_dot)]

    def update(self, thrust_force):
        self.get_acceleration(thrust_force)
        self.eta_dotdot[2] = self.eta_dotdot[2] % np.pi
        self.eta_dot = self.eta_dot + copy.deepcopy(self.eta_dotdot)
        self.eta_dot[2] = self.eta_dot[2] % np.pi
        self.eta = self.eta + copy.deepcopy(self.eta_dot)
        self.eta[2] = self.eta[2] % np.pi
        self.traject.append(copy.deepcopy(self.eta))
        self.vel_traject.append(copy.deepcopy(self.eta_dot))

    def get_current_vel(self, psi):
        if self.V_max >= self.V_c >= self.V_min:
            V_c_dot = np.random.normal(0, 0.01)
        else:
            V_c_dot = -np.random.normal(0, 0.01)
        self.V_c += V_c_dot
        self.V_c = np.clip(self.V_c, self.V_min, self.V_max)
        v_hc = np.asarray([self.V_c * np.cos(-psi), self.V_c * np.sin(-psi), 0]).T
        return v_hc

    def get_acceleration(self, thrust_force):
        # velocity should be the relative velocity
        x, y, psi = self.eta
        u, v, r = self.eta_dot
        m, x_g, y_g = self.m, self.x_g, self.y_g
        gamma_1, gamma_2, gamma_3, gamma_4 = self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4
        M_eta = np.asarray([[m, 0, -m * (y_g * np.cos(psi) + x_g * np.sin(psi))],
                            [0, m, m * (x_g * np.cos(psi) - y_g * np.sin(psi))],
                            [-m * (y_g * np.cos(psi) + x_g * np.sin(psi)), m * (x_g * np.cos(psi) - y_g * np.sin(psi)), self.I_z]])
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
        v_hc = self.get_current_vel(psi)
        last_v_hc = copy.deepcopy(self.v_hc[-1])
        v_hc_dot = (v_hc - last_v_hc)
        v_hr = self.eta_dot - v_hc
        last_v_hr = self.vel_traject[-2] - last_v_hc
        v_hr_dot = (v_hr - last_v_hr)
        self.v_hc.append(v_hc)
        # print(v_hr_dot, v_hr, last_v_hr)

        X_u, Y_v, N_r = -7.2, -7.7, -3
        X_u_dot, Y_v_dot, N_r_dot = -2.9, -3, -3.3
        M_A = -np.diag([X_u_dot, Y_v_dot, N_r_dot])
        C_hA = np.asarray([[0, 0, Y_v_dot * v],
                           [0, 0, -X_u_dot * u],
                           [-Y_v_dot * v, X_u_dot * u, 0]])
        D_h = -np.diag([X_u, Y_v, N_r])
        M_RB = np.asarray([[m, 0, 0],
                           [0, m, 0],
                           [0, 0, self.I_z]])
        C_RB = np.asarray([[0, 0, 0],
                           [0, 0, 0],
                           [m * v, -m * u, 0]])
        d = M_RB @ v_hc_dot.T - M_A @ v_hr_dot.T + C_RB @ v_hc + C_RB @ v_hr - C_hA @ v_hr.T - D_h @ v_hr.T
        # print(d, v_hr, v_hr_dot)
        print(M_RB @ v_hc_dot.T, M_A @ v_hr_dot.T, C_RB @ v_hc, C_RB @ v_hr, C_hA @ v_hr.T, D_h @ v_hr.T)

        M_inv = np.linalg.inv(M_eta)
        J = np.asarray([[np.cos(psi), -np.sin(psi), x],
                        [np.sin(psi), np.cos(psi), y],
                        [0, 0, 1]])
        self.eta_dotdot = -M_inv @ C_eta @ self.eta_dot.T + M_inv @ B_eta @ thrust_force.T + M_inv @ J @ d
        # print(-M_inv @ C_eta @ self.eta_dot.T, M_inv @ np.eye(3) @ d, self.eta_dotdot)

if __name__ == '__main__':
    env = Water()
    thrust_force = np.zeros(4)
    env.reset_robot()
    for _ in range(30):
        env.update(thrust_force)
        # print(env.eta, env.eta_dot, env.eta_dotdot)
    traject = np.asarray(env.traject)
    # plt.plot(traject[0, :], traject[1, :])
    # plt.show()
