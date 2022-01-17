from WaterSim3D import Water
import airsim
import numpy as np


class WaterSim_Airsim:
    def __init__(self, water_env, n_drones):
        self.water_env = water_env
        self.n_drones = n_drones
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        for i in range(n_drones):
            self.client.enableApiControl(True, f"Drone{i + 1}")
            self.client.armDisarm(True, f"Drone{i + 1}")
        for i in range(n_drones):
            f = self.client.takeoffAsync(vehicle_name=f"Drone{i + 1}")
            f.join()
        self.origins = []
        for i in range(n_drones):
            pos = self.client.simGetVehiclePose(vehicle_name=f"Drone{i + 1}").position
            self.origins.append(pos)
        self.step = 0

    def move(self, positions):
        for i in range(self.n_drones):
            pos = positions[i]
            self.client.moveToPositionAsync(pos[0], pos[1], pos[2], 2, vehicle_name=f"Drone{i + 1}").join()
        for i in range(self.n_drones):
            pos = self.client.simGetVehiclePose(vehicle_name=f"Drone{i + 1}").position
            print(pos)

    def update(self, thruster_force):
        self.water_env.update(thruster_force)
        self.step += 1
        if self.step % 15 == 0:
            self.move([self.water_env.eta])


if __name__ == "__main__":
    env = Water()
    watersim = WaterSim_Airsim(env, 1)
    thrust_force = np.zeros(6)
    env.reset_robot()
    for _ in range(60):
        watersim.update(thrust_force)
    traject = np.asarray(env.traject)
    print(traject)