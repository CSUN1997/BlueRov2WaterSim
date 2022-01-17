import numpy as np
import copy
from numpy import sin as s
from numpy import cos as c
import matplotlib.pyplot as plt


class Water:
    def __init__(self, current_model=None):
        self.X_u = -4.03
        self.Y_v = -6.22
        self.Z_w = -5.18
        self.K_p = -0.07
        self.M_q = -0.07
        self.N_r = -0.07
        self.X_u_dot = -5.5
        self.Y_v_dot = -12.7
        self.Z_w_dot = -14.57
        self.K_p_dot = -0.12
        self.M_q_dot = -0.12
        self.N_r_dot = -0.12
        self.V_min = 0
        self.V_max = 0.1
        self.delta_t = 0.1
        self.m = 11.5
        self.W = 112.8
        self.B = 114.8
        self.r_b = np.asarray([0, 0, 0]).T
        self.r_g = np.asarray([0, 0, 0.02])
        self.I_x = 0.16
        self.I_y = 0.16
        self.I_z = 0.16
        self.X_uu = -18.18
        self.Y_vv = -21.66
        self.Z_ww = -36.99
        self.K_pp = -1.55
        self.M_qq = -1.55
        self.N_rr = -1.55

        zg = self.r_g[2]
        self.M_RB = np.asarray([[self.m, 0, 0, 0, self.m * zg, 0],
                                [0, self.m, 0, -self.m * zg, 0, 0],
                                [0, 0, self.m, 0, 0, 0],
                                [0, -self.m * zg, 0, self.I_x, 0, 0],
                                [self.m * zg, 0, 0, 0, self.I_y, 0],
                                [0, 0, 0, 0, 0, self.I_z]])

        self.eta = None
        self.eta_dot = None
        self.eta_dotdot = None
        self.t = None

        self.traject = None
        self.vel_traject = None

        self.V_c = None
        self.v_w = None

        self.reset_robot()

    def get_Jacobian(self):
        phi, theta, psi = self.eta[3:]
        R = np.array([[c(psi) * c(theta), -s(psi) * c(phi) + c(psi) * s(theta) * s(phi), s(psi) * s(phi) + c(psi) * c(phi) * s(theta)],
                      [s(psi) * c(theta), c(psi) * c(phi) + s(phi) * s(theta) * s(psi), -c(psi) * s(phi) + s(theta) * s(psi) * c(phi)],
                      [-s(theta), c(theta) * s(phi), c(theta) * c(phi)]])
        T = np.array([[1, s(phi) * np.tan(theta), c(phi) * np.tan(theta)],
                      [0, c(phi), -s(phi)],
                      [0, s(phi) / c(theta), c(phi) / c(theta)]])
        J = np.block([[R, np.zeros([3, 3])],
                      [np.zeros([3, 3]), T]])
        return J

    def reset_robot(self, x=0.2, y=-0.5, theta=np.pi / 18):
        self.t = 0
        self.V_c = (self.V_max + self.V_min) / 2
        self.eta = np.zeros(6)
        self.eta_dot = np.zeros(6)
        self.eta_dotdot = np.zeros(6)
        self.traject = [np.zeros(6)]
        self.vel_traject = [np.zeros(6)]
        self.v_w = [self.eta_dot - self.get_current_vel()]
        self.v_c = [self.get_current_vel()]

    def update(self, thrust_force):
        self.get_acceleration(thrust_force)
        self.eta_dotdot = self.norm_angles(self.eta_dotdot)
        self.eta_dot = self.norm_angles(self.eta_dot + copy.deepcopy(self.eta_dotdot) * self.delta_t)
        J = self.get_Jacobian()
        self.eta = self.norm_angles(self.eta + J @ copy.deepcopy(self.eta_dot) * self.delta_t)
        self.traject.append(copy.deepcopy(self.eta))
        self.vel_traject.append(copy.deepcopy(self.eta_dot))

    def norm_angles(self, vec):
        vec[3] = vec[3] % np.pi
        vec[4] = vec[4] % (np.pi / 2)
        vec[5] = vec[5] % np.pi
        return vec

    def get_current_vel(self):
        # if self.V_max >= self.V_c >= self.V_min:
        #     V_c_dot = np.random.normal(0, 0.01)
        # else:
        #     V_c_dot = -np.random.normal(0, 0.01)
        # self.V_c += V_c_dot
        # self.V_c = np.clip(self.V_c, self.V_min, self.V_max)
        # v_hc = np.asarray([self.V_c * np.cos(-psi), self.V_c * np.sin(-psi), 0]).T
        # return v_hc
        v = np.zeros(6)
        v[0] = 1
        return v

    def get_C_A(self, velocity):
        u, v, w, p, q, r = velocity
        C_A = np.array([[0, 0, 0, 0, self.Z_w_dot * w, 0],
                        [0, 0, 0, -self.Z_w_dot * w, 0, -self.X_u_dot * u],
                        [0, 0, 0, -self.Y_v_dot * v, self.X_u_dot * u, 0],
                        [0, -self.Z_w_dot * w, self.Y_v_dot * v, 0, -self.N_r_dot * r, -self.M_q_dot * q],
                        [self.Z_w_dot * w, 0, -self.X_u_dot * u, self.N_r_dot * r, 0, -self.K_p_dot * p],
                        [-self.Y_v_dot * v, self.X_u_dot * u, 0, -self.M_q_dot * q, self.K_p_dot * p, 0]])
        return C_A

    def get_D(self, velocity):
        u, v, w, p, q, r = velocity
        D = -np.diag([self.X_u + self.X_uu * np.abs(u), self.Y_v + self.Y_vv * np.abs(v),
                      self.Z_w + self.Z_ww * np.abs(w), self.K_p + self.K_pp * np.abs(p),
                      self.M_q + self.M_qq * np.abs(q), self.N_r + self.N_rr * np.abs(r)])
        return D

    def get_g(self):
        state = self.eta
        x, y, z, phi, theta, psi = state
        g = np.asarray([(self.W - self.B) * s(theta), -(self.W - self.B) * c(theta) * s(phi),
                        -(self.W - self.B) * c(theta) * c(phi), self.r_g[2] * self.W * c(theta) * s(phi),
                        self.r_g[2] * self.W * s(theta), 0]).T
        return g

    def get_acceleration(self, thrust_force):
        # velocity should be the relative velocity
        u, v, w, p, q, r = self.eta_dot
        m = self.m
        C_RB = np.array([[0, 0, 0, 0, m * w, 0],
                         [0, 0, 0, -m * w, 0, 0],
                         [0, 0, 0, m * v, -m * u, 0],
                         [0, m * w, -m * v, 0, self.I_z * r, -self.I_y * q],
                         [-m * w, 0, -m * u, -self.I_z * r, 0, self.I_x * p],
                         [m * v, -m * u, 0, self.I_y * q, -self.I_x * p, 0]])
        M_A = -np.diag([self.X_u_dot, self.Y_v_dot, self.Z_w_dot, self.K_p_dot, self.M_q_dot, self.N_r_dot])
        v_c = self.get_current_vel()
        v_w = self.eta_dot - v_c
        v_w_dot = (v_w - self.v_w[-1]) / self.delta_t
        # print(v_w, self.v_w[-1])
        v_dot = self.eta_dotdot
        C_A = self.get_C_A(v_w)
        D = self.get_D(v_w)
        g = self.get_g()
        # part1 = self.M_RB @ v_dot.T
        # part2 = C_RB @ self.eta_dot.T
        # part3 = M_A @ v_w_dot.T
        # part4 = C_A @ v_w.T
        # part5 = D @ v_w.T
        # J = self.get_Jacobian()
        v_c_dot = (v_c - self.v_c[-1]) / self.delta_t
        part1 = self.M_RB @ v_c_dot
        part2 = -M_A @ v_w_dot
        part3 = C_RB @ v_c
        part4 = C_RB @ v_w
        part5 = -C_A @ v_w
        part6 = -D @ v_w

        tau = part1 + part2 + part3 + part4 + part5 + part6
        # print(1, part1)
        # print(2, part2)
        # print(3, part3)
        # print(4, part4)
        # print(5, part5)
        # print(6, g)
        # print(tau)
        self.eta_dotdot = tau / self.m
        self.v_w.append(v_w)
        self.v_c.append(v_c)


if __name__ == '__main__':
    env = Water()
    thrust_force = np.zeros(4)
    env.reset_robot()
    for _ in range(300):
        env.update(thrust_force)
        print(env.eta_dot, env.eta_dotdot)
    traject = np.asarray(env.traject)
    print(traject[:, 0])
    plt.scatter(traject[:, 0], traject[:, 1])
    plt.show()
