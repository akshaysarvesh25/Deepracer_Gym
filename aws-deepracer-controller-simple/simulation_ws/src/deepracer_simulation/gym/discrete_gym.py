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

class DeepracerDiscreteGym(gym.Env):

	def __init__(self,target_point):
		super(DeepracerGym,self).__init__()
		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		#self.action_space = spaces.Discrete(n_actions)
		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32) # speed and steering
		# self.pose_observation_space = spaces.Box(np.array([-1. , -1., -1.]),np.array([1., 1., 1.]),dtype = np.float32)
		# self.lidar_observation_space = spaces.Box(0,1.,shape=(720,),dtype = np.float32)
		# self.observation_space = spaces.Tuple((self.pose_observation_space,self.lidar_observation_space))
		low = np.concatenate((np.array([-1.,-1.,-4.]),np.zeros(8)))
		high = np.concatenate((np.array([1.,1.,4.]),np.zeros(8)))
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.target_point_ = np.array([target_point[0]/MAX_X,target_point[1]/MAX_Y])
		#self.lidar_ranges_ = np.zeros(720)
		self.temp_lidar_values_old = np.zeros(8)
	
	def reset(self):        
		global yaw_car, lidar_range_values
		#time.sleep(1e-2)
		self.stop_car()        
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			# pause physics
			# reset simulation
			# un-pause physics
			self.pause()
			self.reset_simulation_proxy()
			self.unpause()
			print('Simulation reset')
		except rospy.ServiceException as exc:
			print("Reset Service did not process request: " + str(exc))

		pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car],dtype=np.float32) #relative pose 
		temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		temp_lidar_values = temp_lidar_values/max_lidar_value
		print(temp_lidar_values.shape)
		temp_lidar_values = np.mean(temp_lidar_values.reshape(-1,45), axis = 1)
		print(temp_lidar_values.shape)
		return_state = np.concatenate((pose_deepracer,temp_lidar_values))
		
		# if ((max(return_state) > 1.) or (min(return_state < -1.)) or (len(return_state) != 723)):
		# 	print('-----------------ERROR Reset----------------------')        
		
		return return_state
	
	def get_reward(self,x,y):
		x_target = self.target_point_[0]
		y_target = self.target_point_[1]
		head = math.atan((self.target_point_[1]-y)/(self.target_point_[0]-x+0.01))
		return -1*(abs(x - x_target) + abs(y - y_target) + abs (head - yaw_car)) # reward is -1*distance to target, limited to [-1,0]

	def step(self,action):
		global yaw_car, lidar_range_values
		# self.lidar_ranges_ = np.array(lidar_range_values)
		self.temp_lidar_values_old = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		self.temp_lidar_values_old = self.temp_lidar_values_old/max_lidar_value
		self.temp_lidar_values_old = np.mean(self.temp_lidar_values_old.reshape(-1,45), axis = 1)

		global x_pub
		msg = AckermannDriveStamped()
		msg.drive.speed = action[0]*MAX_VEL
		msg.drive.steering_angle = action[1]*MAX_STEER
		x_pub.publish(msg)

		reward = 0
		done = False

		if((abs(pos[0]) < 1.) and (abs(pos[1]) < 1.) ):

			if(min(self.temp_lidar_values_old)<0.4/max_lidar_value):
				reward = -1 + self.get_reward(pos[0],pos[1])   
				done = False
			
			elif(abs(pos[0]-self.target_point_[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(pos[1]-self.target_point_[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')

			else:
				reward = self.get_reward(pos[0],pos[1])

			pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car],dtype=np.float32) #relative pose

		else: 
			done = True
			print('Outside Range')
			reward = -1
			temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
			temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]

			head = math.atan((self.target_point_[1]-pos[1])/(self.target_point_[0]-pos[0]+0.01)) #calculate pose to target dierction
			pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car],dtype=np.float32) #relative pose 

		info = {}

		# self.lidar_ranges_ = np.array(lidar_range_values)
		
		temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		temp_lidar_values = temp_lidar_values/max_lidar_value
		temp_lidar_values = np.mean(temp_lidar_values.reshape(-1,45), axis = 1)

		return_state = np.concatenate((pose_deepracer,temp_lidar_values))
		
		# if ((max(return_state) > 1.) or (min(return_state < -1.)) or (len(return_state) != 723)):
		# 	print('-----------------ERROR Step----------------------')
		# 	print(max(pose_deepracer),max(temp_lidar_values))
		# 	print(min(pose_dseepracer),min(temp_lidar_values))
		# 	print(len(return_state))
		# 	print('-------------------------------------------------')

		return return_state,reward,done,info     

	def stop_car(self):
		global x_pub
		msg = AckermannDriveStamped()
		msg.drive.speed = 0.
		msg.drive.steering_angle = 0.
		x_pub.publish(msg)
	
	def render(self):
		pass

	def close(self):
		pass