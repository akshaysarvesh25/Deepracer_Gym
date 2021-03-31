#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
from std_srvs.srv import Empty
import argparse
import datetime
import itertools
import torch, gc
import message_filters
gc.collect()

from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

MAX_X = 20
MAX_Y = 20
pos = [0,0]
yaw_car = 0

def euler_from_quaternion(x, y, z, w):
	"""
	Convert a quaternion into euler angles (roll, pitch, yaw)
	roll is rotation around x in radians (counterclockwise)
	pitch is rotation around y in radians (counterclockwise)
	yaw is rotation around z in radians (counterclockwise)
	"""
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)

	return roll_x, pitch_y, yaw_z # in radians

def get_vehicle_state(data):
	pass

def get_lidar_data(data):
	# global lidar_range_values
	# lidar_range_values = np.array(data.ranges,dtype=np.float32)
	pass

def filtered_data(pose_data,lidar_data):
	global pos,velocity,old_pos
	racecar_pose = pose_data.pose[2]
	pos[0] = racecar_pose.position.x/MAX_X
	pos[1] = racecar_pose.position.y/MAX_Y
	quaternion = (
			pose_data.pose[2].orientation.x,
			pose_data.pose[2].orientation.y,
			pose_data.pose[2].orientation.z,
			pose_data.pose[2].orientation.w)
	q = quaternion
	euler =  euler_from_quaternion(q[0],q[1],q[2],q[3])
	yaw = euler[2]
	yaw_car = yaw

	global lidar_range_values
	lidar_range_values = np.array(lidar_data.ranges,dtype=np.float32)
	print(pos, yaw)
	# print(lidar_range_values)

	pass

def start():
	torch.cuda.empty_cache()

	rospy.init_node('deepracer_controller_mpc', anonymous=True)
	
	pose_sub2 = rospy.Subscriber("/gazebo/model_states_drop",ModelStates,get_vehicle_state)
	# x_sub1 = rospy.Subscriber("/move_base_simple/goal",PoseStamped,get_clicked_point)
	lidar_sub2 = rospy.Subscriber("/scan", LaserScan, get_lidar_data)
	pose_sub = message_filters.Subscriber("/gazebo/model_states_drop", ModelStates)
	lidar_sub = message_filters.Subscriber("/scan", LaserScan)
	ts = message_filters.ApproximateTimeSynchronizer([pose_sub,lidar_sub],10,0.1,allow_headerless=True)
	ts.registerCallback(filtered_data)

	rospy.spin()

if __name__ == '__main__':
	try:
		Flag = False
		Flag = start()

		if Flag:
			print('----------_All Done-------------')
	except rospy.ROSInterruptException:
		pass