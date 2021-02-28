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
from stable_baselines.common.env_checker import check_env



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
        self.pose_observation_space = spaces.Box(np.array([-1. , -1., -1.]),np.array([1., 1., 1.]),dtype = np.float32)
        self.lidar_observation_space = spaces.Box(0,1.,shape=(720,),dtype = np.float32)
        self.observation_space = spaces.Tuple((self.pose_observation_space,self.lidar_observation_space))
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.target_point_ = np.array([target_point[0]/MAX_X,target_point[1]/MAX_Y])
        self.lidar_ranges_ = np.zeros(720)
    
    def reset(self):
        global yaw_car
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
            print('Simulation reset')
        except rospy.ServiceException as exc:
            print("Reset Service did not process request: " + str(exc))
        pose_deepracer = np.array([pos[0],pos[1],yaw_car],dtype=np.float32)        
        
        return (pose_deepracer,self.lidar_ranges_)

    
    def get_reward(self,x,y):
        x_target = self.target_point_[0]
        y_target = self.target_point_[1]
        return max(-1, -1*np.sqrt((x - x_target)**2 + (y - y_target)**2)) # reward is -1*distance to target, limited to [-1,0]




    def step(self,action):
        global yaw_car
        # print('Taking action ')
        # print('Changing velocity to : ',action[0],' m/s')
        # print('Changing steering angle to : ',action[1],' radians')
        global x_pub
        msg = AckermannDriveStamped()
        msg.drive.speed = action[0]*MAX_VEL
        msg.drive.steering_angle = action[1]*MAX_STEER
        x_pub.publish(msg)
        reward = 0
        done = False

        if((abs(pos[0]) < 1.) and (abs(pos[1]) < 1.) ):
            reward = self.get_reward(pos[0],pos[1])

            if(abs(pos[0]-self.target_point_[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(pos[1]-self.target_point_[1])<THRESHOLD_DISTANCE_2_GOAL):
                reward = 1            
                done = True
                print('Goal Reached')                
           
            self.lidar_ranges_ = np.array(lidar_range_values)            
            if(min(self.lidar_ranges_)<0.4/max_lidar_value):
                reward = -1            
                done = True
                print('Simulation reset because of collission')

            pose_deepracer = np.array([pos[0],pos[1],yaw_car])

        else: 
            done = True
            print('Outside Range')
            reward = -1
            temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
            temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]
            pose_deepracer = np.array([temp_pos0, temp_pos1, yaw_car])
        
        

        info = {}

        return (pose_deepracer,self.lidar_ranges_),reward,done,info
        
    
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
    yaw_car = yaw/MAX_YAW
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
    
    # normalized_ranges = []
    # for vals in data.ranges:
    #     normalized_ranges.append(vals/14)
    #     if(vals>=12.00):
    #         normalized_ranges.append(1)

    lidar_range_values = np.array(data.ranges,dtype=np.float32)
    lidar_range_values = np.nan_to_num(lidar_range_values, copy=True, posinf=max_lidar_value)
    lidar_range_values = lidar_range_values/max_lidar_value
    # print(lidar_range_values)




def start():

    rospy.init_node('deepracer_controller_mpc', anonymous=True)
    
    x=rospy.Subscriber("/gazebo/model_states_drop",ModelStates,get_vehicle_state)
    x_sub1 = rospy.Subscriber("/move_base_simple/goal",PoseStamped,get_clicked_point)
    x_sub2 = rospy.Subscriber("/scan",LaserScan,get_lidar_data)
    target_point = [0,0]
    env =  DeepracerGym(target_point)
    # max_time_step = 1000000
    max_eposide = 100
    e = 0
    while not rospy.is_shutdown():
        time.sleep(1)        
        while(e < max_eposide):
            e += 1      
            state = env.reset()
            Flag = False                             
            while(~Flag):
                action = np.array([0.1,0.0])
                n_state,reward,done,info = env.step(action)
                print(reward)
                if done:                   
                    Flag = True
                    break
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
