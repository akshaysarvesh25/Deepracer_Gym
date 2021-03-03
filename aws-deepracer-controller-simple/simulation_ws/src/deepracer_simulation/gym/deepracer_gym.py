#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
from gazebo_msgs.msg import ModelStates
#import tf
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
#from cvxpy import *
#import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
from std_srvs.srv import Empty

# from stable_baselines.common.env_checker import check_env
# from stable_baselines.td3.policies import MlpPolicy
# from stable_baselines import SAC,TD3
import argparse
import datetime
import itertools
import torch
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
# from stable_baselines3 import SAC
# from stable_baselines3.sac import MlpPolicy


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
					help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
					help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
					help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
					help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
					help='Temperature parameter α determines the relative importance of the entropy\
							term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
					help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
					help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
					help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
					help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
					help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
					help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
					help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
					help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
					help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
					help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=3000, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()

x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)
pos = [0,0]
yaw_car = 0
MAX_VEL = 10.
steer_precision = 0#1e-3
MAX_STEER = (np.pi*0.25) - steer_precision
MAX_YAW = 2*np.pi
MAX_X = 20
MAX_Y = 20
# target_x = 50/MAX_X
# target_y = 50/MAX_Y
max_lidar_value = 14
# target_point = [target_x,target_y]
THRESHOLD_DISTANCE_2_GOAL = 0.6/max(MAX_X,MAX_Y)

class DeepracerGym(gym.Env):

	def __init__(self,target_point):
		super(DeepracerGym,self).__init__()
		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		#self.action_space = spaces.Discrete(n_actions)
		self.action_space = spaces.Box(np.array([-1., -1.]), np.array([1., 1.]), dtype = np.float32)
		# self.pose_observation_space = spaces.Box(np.array([-1. , -1., -1.]),np.array([1., 1., 1.]),dtype = np.float32)
		# self.lidar_observation_space = spaces.Box(0,1.,shape=(720,),dtype = np.float32)
		# self.observation_space = spaces.Tuple((self.pose_observation_space,self.lidar_observation_space))
		low = np.concatenate((np.array([-1.,-1.,-4.]),np.zeros(720)))
		high = np.concatenate((np.array([1.,1.,4.]),np.zeros(720)))
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.target_point_ = np.array([target_point[0]/MAX_X,target_point[1]/MAX_Y])
		#self.lidar_ranges_ = np.zeros(720)
		self.temp_lidar_values_old = np.zeros(720)
	
	def reset(self):        
		global yaw_car
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

		head = math.atan((self.target_point_[1]-pos[1])/(self.target_point_[0]-pos[0]+0.01))

		pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), abs(head - yaw_car)],dtype=np.float32) #relative pose and yaw

		temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		temp_lidar_values = temp_lidar_values/max_lidar_value
		return_state = np.concatenate((pose_deepracer,temp_lidar_values))
		
		if ((max(return_state) > 1.) or (min(return_state < -1.)) or (len(return_state) != 723)):
			print('-----------------ERROR Reset----------------------')        
		
		return return_state
	
	def get_reward(self,x,y):
		x_target = self.target_point_[0]
		y_target = self.target_point_[1]
		return -1*(abs(x - x_target) + abs(y - y_target)) # reward is -1*distance to target, limited to [-1,0]

	def step(self,action):
		global yaw_car
		# self.lidar_ranges_ = np.array(lidar_range_values)
		self.temp_lidar_values_old = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		self.temp_lidar_values_old = self.temp_lidar_values_old/max_lidar_value

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
				# print('Collission')

			
			elif(abs(pos[0]-self.target_point_[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(pos[1]-self.target_point_[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')

			else:
				reward = self.get_reward(pos[0],pos[1])

			head = math.atan((self.target_point_[1]-pos[1])/(self.target_point_[0]-pos[0]+0.01))

			pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]),abs(head - yaw_car)],dtype=np.float32) #relative pose and yaw

		else: 
			done = True
			print('Outside Range')
			reward = -1
			temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
			temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]

			head = math.atan((self.target_point_[1]-pos[1])/(self.target_point_[0]-pos[0]+0.01))
			pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), abs(head - yaw_car)],dtype=np.float32) #relative pose and yaw   

		info = {}

		# self.lidar_ranges_ = np.array(lidar_range_values)
		temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		temp_lidar_values = temp_lidar_values/max_lidar_value


		return_state = np.concatenate((pose_deepracer,temp_lidar_values))
		if ((max(return_state) > 1.) or (min(return_state < -1.)) or (len(return_state) != 723)):
			print('-----------------ERROR Step----------------------')
			print(max(pose_deepracer),max(temp_lidar_values))
			print(min(pose_deepracer),min(temp_lidar_values))
			print(len(return_state))
			print('-------------------------------------------------')
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

lidar_range_values = []
origin = [0,0]
count = 0
DISPLAY_COUNT=10
WB = 0.15 #[m]
DT = 1
velocity = 0
old_pos = [0,0]
delta = 0
T_Horizon = 5
n = 3
m = 2
R = np.diag([0.01, 1.0])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5])  # state cost matrix

class State:
	def __init__(self, x=0.0, y=0.0, yaw=0.0):
		self.x = x
		self.y = y
		self.yaw = yaw
		self.predelta = None

def get_clicked_point(data):
	print("Clicked point : ",data)
	"""
	global target_point
	target = [data.pose.position.x,data.pose.position.y]
	target_point[0] = data.pose.position.x
	target_point[1] = data.pose.position.y
	print("Target Point : ",target_point)

	point1 = [pos[0],pos[1]]
	point2 = target_point
	x,y = hanging_line_display(point1, point2)
	#print(x,y)
	global count
	count = count+1

	#x,y = discretize_points(x,y)
	x = x[1::10]
	y = y[1::10]

	if(count<=DISPLAY_COUNT):
		plt.plot(point1[0], point1[1], 'o')
		plt.plot(point2[0], point2[1], 'o')
		plt.plot(x,y,'.')
		plt.show()
	"""



#https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
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
	
	global pos,velocity,old_pos
	racecar_pose = data.pose[2]
	pos[0] = racecar_pose.position.x/MAX_X
	pos[1] = racecar_pose.position.y/MAX_Y
	quaternion = (
			data.pose[2].orientation.x,
			data.pose[2].orientation.y,
			data.pose[2].orientation.z,
			data.pose[2].orientation.w)
	#euler = tf.transformations.euler_from_quaternion(quaternion)
	q = quaternion
	euler =  euler_from_quaternion(q[0],q[1],q[2],q[3])
	yaw = euler[2]
	yaw_car = yaw
	velocity = get_current_velocity(old_pos[0],old_pos[1],pos[0],pos[1])
	# print(pos[0],pos[1],yaw)
	#print('velocity = ',velocity)
	#old_pos[0] = pos[0]
	#old_pos[1] = pos[1]
	#state = State(x=pos[0], y=pos[1], yaw=yaw)

	#print("Car is at :",pos[0], pos[1], yaw)
	#pointX,pointY = get_trajectory(pos[0],pos[1])
	#print('Trajectory points : ',pointX,pointY)
	#print('First target point : ',pointX[2],pointY[2])

	#linear_mpc_control(pointX[2],pointY[2],pos[0],pos[1],yaw)

	#x,y,theta = update_vehicle_predicted_state(pos[0],pos[1],yaw)
	#print('x,y,phi',x,y,theta)



	#print(pointX,pointY)





def get_current_velocity(x_old,y_old,x_new,y_new):
	vel = math.sqrt(pow((x_old-x_new),2)+pow((x_old-x_new),2))
	return vel

def get_lidar_data(data):
	#print(type(data.ranges))
	# i = 1+1
	global lidar_range_values
	lidar_range_values = np.array(data.ranges,dtype=np.float32)
	# if len(lidar_range_values) < 720:
	# 	lidar_range_values = np.zeros(720)

	# normalized_ranges = []
	# for vals in data.ranges:
	#     normalized_ranges.append(vals/14)
	#     if(vals>=12.00):
	#         normalized_ranges.append(1)


	# print(lidar_range_values)




def start():

	rospy.init_node('deepracer_controller_mpc', anonymous=True)
	
	x=rospy.Subscriber("/gazebo/model_states_drop",ModelStates,get_vehicle_state)
	x_sub1 = rospy.Subscriber("/move_base_simple/goal",PoseStamped,get_clicked_point)
	x_sub2 = rospy.Subscriber("/scan",LaserScan,get_lidar_data)
	target_point = [-2,0]
	env =  DeepracerGym(target_point)
	'''
	while not rospy.is_shutdown():
		time.sleep(1)
		print('---------------------------',check_env(env))
	'''

	
	##
	
	# max_time_step = 100000
	# max_eposide = 1000
	# e = 0
	# while not rospy.is_shutdown():
	# 	time.sleep(1) #Do not remove this 
	# 	state = env.reset()        
	# 	while(e < max_eposide):
	# 		e += 1  
	# 		# state = env.reset()          
	# 		for _ in range(max_time_step):
	# 			action = np.array([0.5,0.1])
	# 			n_state,reward,done,info = env.step(action)
	# 			# print(reward)
	# 			if done:
	# 				state = env.reset()                   
	# 				break
	
	# rospy.spin()

	


	while not rospy.is_shutdown():
		time.sleep(1) #Do not remove this 
		state = env.reset() #Do not remove this 
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)

		# Agent
		agent = SAC(env.observation_space.shape[0], env.action_space, args)
		# Memory
		memory = ReplayMemory(args.replay_size, args.seed)
		#Tesnorboard
		writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'DeepracerGym',
															 args.policy, "autotune" if args.automatic_entropy_tuning else ""))
		total_numsteps = 0
		updates = 0
		num_goal_reached = 0

		for i_episode in itertools.count(1):
			episode_reward = 0
			episode_steps = 0
			done = False
			state = env.reset()
			


			while not done:
				
				if args.start_steps > total_numsteps:
					action = env.action_space.sample()  # Sample random action
				else:
					action = agent.select_action(state)  # Sample action from policy

				if len(memory) > args.batch_size:
					# Number of updates per step in environment
					for i in range(args.updates_per_step):
						# Update parameters of all the networks
						critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

						writer.add_scalar('loss/critic_1', critic_1_loss, updates)
						writer.add_scalar('loss/critic_2', critic_2_loss, updates)
						writer.add_scalar('loss/policy', policy_loss, updates)
						writer.add_scalar('loss/entropy_loss', ent_loss, updates)
						writer.add_scalar('entropy_temprature/alpha', alpha, updates)
						updates += 1

				next_state, reward, done, _ = env.step(action) # Step
				if reward > 9: #Count the number of times the goal is reached
					num_goal_reached += 1

				episode_steps += 1
				total_numsteps += 1
				episode_reward += reward
				if episode_steps > args.max_episode_length:
					done = True

				# Ignore the "done" signal if it comes from hitting the time horizon.
				# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
				mask = 1 if episode_steps == args.max_episode_length else float(not done)
				# mask = float(not done)
				memory.push(state, action, reward, next_state, mask) # Append transition to memory

				state = next_state

			if total_numsteps > args.num_steps:
				break

			writer.add_scalar('reward/train', episode_reward, i_episode)
			writer.add_scalar('reward/episode_length',episode_steps, i_episode)
			writer.add_scalar('reward/num_goal_reached',num_goal_reached, i_episode)
			print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
			print("Number of Goals Reached: ",num_goal_reached)

		print('----------------------Training Ending----------------------')
		return True


		
	rospy.spin()
	

if __name__ == '__main__':
	try:
		Flag = False
		Flag = start()
		if Flag:
			print('----------_All Done-------------')
	except rospy.ROSInterruptException:
		pass
