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
import random
import csv

from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
					help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
					help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
					help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
					help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
					help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
					help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
					help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda',type=int, default=1, metavar='N',
					help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=200, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()
rospy.init_node('deepracer_gym', anonymous=True)
x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)

pos = [0,0]
old_pos = [0,0]
yaw_car = 0
MAX_VEL = 1.0
steer_precision = 0 # 1e-3
MAX_STEER = (np.pi/2.0) - steer_precision
MAX_YAW = 2*np.pi
MAX_X = 5
MAX_Y = 5
THRESHOLD_DISTANCE_2_GOAL = 0.25/max(MAX_X,MAX_Y)
UPDATE_EVERY = 5
count = 0
total_numsteps = 0
updates = 0
num_goal_reached = 0
done = False
i_episode = 1
episode_reward = 0
max_ep_reward = 0
episode_steps = 0
memory = ReplayMemory(args.replay_size, args.seed)

class DeepracerGym(gym.Env):

	def __init__(self):
		super(DeepracerGym,self).__init__()
		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32) # speed and steering
		low = np.concatenate((np.array([-1.,-1.,-4.]),np.zeros(8)))
		high = np.concatenate((np.array([1.,1.,4.]),np.zeros(8)))
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.target_point = np.array([0./MAX_X, 0./MAX_Y])
		self.pose = np.array([pos[0]/MAX_X, pos[1]/MAX_Y, yaw_car])
	
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

		pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose 
	
		return_state = pose_deepracer

		random_targets = [[-1., -1.], [-0., -1.], [1., -1.]]
		target_point = random.choice(random_targets)
		self.target_point = np.array([target_point[0]/MAX_X,target_point[1]/MAX_Y])
		print("Episode Target Point : ", self.target_point)  
		return return_state

	def get_distance(self,x1,x2):
		# Distance between points x1 and x2
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1,x2):
		# Heading between points x1,x2 with +X axis
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))	
	
	def get_reward(self,x,y):
		x_target = self.target_point[0]
		y_target = self.target_point[1]
		x = self.pose[0]
		y = self.pose[1]
		head_to_target = self.get_heading(self.pose, self.target_point)
		alpha = head_to_target - self.pose[2]
		ld = self.get_distance(self.pose, self.target_point)
		crossTrackError = math.sin(alpha) *ld
		return -1*(abs(crossTrackError)**2 + abs(x - x_target) + abs(y - y_target) + 3*abs (head_to_target - yaw_car)/1.57)/6 # reward is -1*distance to target, limited to [-1,0]

	def step(self,action):
		global yaw_car
		global x_pub
		msg = AckermannDriveStamped()
		msg.drive.speed = action[0]*MAX_VEL
		msg.drive.steering_angle = action[1]*MAX_STEER
		x_pub.publish(msg)

		reward = 0
		done = False

		# pose_csv = [5*pos[0], 5*pos[1]]
		# with open('poses.csv', 'a', newline = '') as csvFile:
		# 	writer = csv.writer(csvFile)
		# 	writer.writerow(pose_csv)
		# 	csvFile.close()

		if((abs(pos[0]) < 1.) and (abs(pos[1]) < 1.) ):
			
			if(abs(pos[0]-self.target_point[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(pos[1]-self.target_point[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')

			else:
				reward = self.get_reward(pos[0],pos[1])

			pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose

		else: 
			done = True
			print('Outside Range')
			reward = -1
			temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
			temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]

			head = math.atan((self.target_point[1]-pos[1])/(self.target_point[0]-pos[0]+0.01)) #calculate pose to target dierction
			pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose 

		info = {}

		return_state = pose_deepracer

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

target_point = [-3.0, -3.]
env =  DeepracerGym()
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'DeepracerGym',
															 args.policy, "autotune" if args.automatic_entropy_tuning else ""))
# actor_path = "models/sac_actor_random_gazebo_1"
# critic_path = "models/sac_critic_random_gazebo_1"
agent = SAC(env.observation_space.shape[0], env.action_space, args)
# agent.load_model(actor_path, critic_path)

state = np.zeros(env.observation_space.shape[0])

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def network_update():
	global updates, episode_reward, episode_steps, num_goal_reached, i_episode
	if len(memory) > args.batch_size:
		# Number of updates per step in environment
		for i in range(args.updates_per_step*args.max_episode_length):
			# Update parameters of all the networks
			critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
			writer.add_scalar('loss/critic_1', critic_1_loss, updates)
			writer.add_scalar('loss/critic_2', critic_2_loss, updates)
			writer.add_scalar('loss/policy', policy_loss, updates)
			writer.add_scalar('loss/entropy_loss', ent_loss, updates)
			writer.add_scalar('entropy_temprature/alpha', alpha, updates)
			updates += 1

		if (episode_steps > 1):
			writer.add_scalar('reward/train', episode_reward, i_episode)
			writer.add_scalar('reward/episode_length',episode_steps, i_episode)
			writer.add_scalar('reward/num_goal_reached',num_goal_reached, i_episode)

		print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
		print("Number of Goals Reached: ",num_goal_reached)

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

def pose_callback(pose_data):
	global pos,velocity,old_pos, total_numsteps, done, env, episode_steps, episode_reward, memory, state, ts, x_pub, num_goal_reached, i_episode
	global updates, episode_reward, episode_steps, num_goal_reached, i_episode, max_ep_reward
	racecar_pose = pose_data.pose[1]
	pos[0] = racecar_pose.position.x/MAX_X
	pos[1] = racecar_pose.position.y/MAX_Y
	q = (
			pose_data.pose[1].orientation.x,
			pose_data.pose[1].orientation.y,
			pose_data.pose[1].orientation.z,
			pose_data.pose[1].orientation.w)
	euler =  euler_from_quaternion(q[0],q[1],q[2],q[3])
	yaw = euler[2]
	yaw_car = yaw

	if total_numsteps > args.num_steps:
		print('----------------------Training Ending----------------------')
		env.stop_car()			
		agent.save_model("random_gazebo", suffix = "1")
		ts.unregister()
		pass

	if not done:

		if args.start_steps > total_numsteps:
			action = env.action_space.sample()  # Sample random action
		else:
			action = agent.select_action(state)  # Sample action from policy	

		next_state, reward, done, _ = env.step(action) # Step
		rospy.sleep(0.02)

		if (reward > 9) and (episode_steps > 1): #Count the number of times the goal is reached
			num_goal_reached += 1 

		episode_steps += 1
		total_numsteps += 1
		episode_reward += reward

		if episode_steps > args.max_episode_length:
			done = True

		print(episode_steps, end = '\r')
		# Ignore the "done" signal if it comes from hitting the time horizon.
		# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
		mask = 1 if episode_steps == args.max_episode_length else float(not done)
		# mask = float(not done)
		memory.push(state, action, reward, next_state, mask) # Append transition to memory
		state = next_state
	else:
		state = env.reset()
		# network_update()
		i_episode += 1
		if episode_reward >= max_ep_reward:
			max_ep_reward = episode_reward
			print("Saving checkpoint model")
			# agent.save_model("checkpoint", suffix = "1")
		episode_reward = 0
		episode_steps = 0
		done = False

def start():
	global ts
	torch.cuda.empty_cache()	
	rospy.init_node('deepracer_gym', anonymous=True)		
	pose_sub = rospy.Subscriber("/gazebo/model_states_drop", ModelStates, pose_callback)
	state = env.reset()
	rospy.spin()

if __name__ == '__main__':
	try:
		Flag = False
		Flag = start()
		if Flag:
			print('----------_All Done-------------')
	except rospy.ROSInterruptException:
		pass