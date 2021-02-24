#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
from gazebo_msgs.msg import ModelStates
import tf
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
#from stable_baselines.common.env_checker import check_env



x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)
pos = [0,0]
target_point = [5,5]
yaw_car = 0
THRESHOLD_DISTANCE_2_GOAL=0.6


class DeepracerGym(gym.Env):

    def __init__(self):
        super(DeepracerGym,self).__init__()
        
        n_actions = 2 #velocity,steering
        metadata = {'render.modes': ['console']}
        #self.action_space = spaces.Discrete(n_actions)
        self.action_space = spaces.Box(np.array([-10, -10]), np.array([10, 10]), dtype = np.float32)
        self.pose_observation_space = spaces.Box(np.array([-10000,-10000,-6.3]),np.array([10000,10000,6.3]),dtype = np.float32)
        self.lidar_observation_space = spaces.Box(np.array([0]),np.array([13]),dtype = np.float32)
        self.observation_space = spaces.Tuple([self.pose_observation_space,self.lidar_observation_space])
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.target_point_ = target_point
        self.lidar_ranges_ = np.array(lidar_range_values)
    
    def reset(self):
        global yaw_car
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
            print('Simulation reset')
        except rospy.ServiceException as exc:
            print("Reset Service did not process request: " + str(exc))
        pose_deepracer = np.array([pos[0],pos[1],yaw_car])


        return [pose_deepracer,self.lidar_ranges_]

    def step(self,action):
        global yaw_car
        print('Taking action ')
        print('Changing velocity to : ',action[0],' m/s')
        print('Changing steering angle to : ',action[1],' radians')
        global x_pub
        msg = AckermannDriveStamped()
        msg.drive.speed = action[0]
        msg.drive.steering_angle = action[1]
        x_pub.publish(msg)
        reward = 0
        done = 0
        if(abs(pos[0]-self.target_point_[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(pos[1]-self.target_point_[1])<THRESHOLD_DISTANCE_2_GOAL):
            reward = 1000
            done = 1
        print(self.lidar_ranges_)
        self.lidar_ranges_ = np.array(lidar_range_values)
        print(self.lidar_ranges_)
        
        if(min(self.lidar_ranges_)<0.4):
            reward = -1000
            done = 0
            print('Simulation reset because of collission')
        
        pose_deepracer = np.array([pos[0],pos[1],yaw_car])

        info = {}

        return [pose_deepracer,self.lidar_ranges_],reward,done,info
        
    
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


def get_vehicle_state(data):
    
    global pos,velocity,old_pos
    racecar_pose = data.pose[2]
    pos[0] = racecar_pose.position.x
    pos[1] = racecar_pose.position.y
    quaternion = (
            data.pose[2].orientation.x,
            data.pose[2].orientation.y,
            data.pose[2].orientation.z,
            data.pose[2].orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]
    yaw_car = yaw
    velocity = get_current_velocity(old_pos[0],old_pos[1],pos[0],pos[1])
    print(pos[0],pos[1],yaw)
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
    i = 1+1
    global lidar_range_values
    lidar_range_values = np.array(data.ranges)
    #print(lidar_range_values)




def start():

    rospy.init_node('deepracer_controller_mpc', anonymous=True)
    
    x=rospy.Subscriber("/gazebo/model_states_drop",ModelStates,get_vehicle_state)
    x_sub1 = rospy.Subscriber("/move_base_simple/goal",PoseStamped,get_clicked_point)
    x_sub2 = rospy.Subscriber("/scan",LaserScan,get_lidar_data)
    env =  DeepracerGym()
    while not rospy.is_shutdown():
        time.sleep(1)
        #obs = env.reset()
        action = np.array([5,1.2])
        x, reward, done, info = env.step(action)
        print(x[0][0])
        print(reward)

        """


        Write your code here



        <end code>
        """
    rospy.spin()

if __name__ == '__main__':
    try:
        start()
    except rospy.ROSInterruptException:
        pass
