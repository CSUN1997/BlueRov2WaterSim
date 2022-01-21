import airsim
import cv2
import numpy as np
import os
import pprint
import tempfile
# connect to the AirSim simulator
# client = airsim.MultirotorClient()
client = airsim.VehicleClient()
client.confirmConnection()
# client.enableApiControl(True, "Drone1")
# client.enableApiControl(True, "Drone2")
# client.armDisarm(True, "Drone1")
# client.armDisarm(True, "Drone2")

client.simAddVehicle("Drone3", "simpleflight", [0, 0, 0], "D:\\BlueROV2\\ROV.fbx")
