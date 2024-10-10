"""
    这个文件是踢狗人的控制器, 需要定义一些动作来完成对 虚拟环境下 预训练数据的构造

    大臂举平 是 -1.6到1.6

    小臂向上弯曲到30度角,到向下打平 是-2 到0

    大腿前后 举平 是 -1.5 到1.5

    小腿向后弯曲到与大腿30度夹角 0到2.0

    头可以抬头和低头

    脚 是 -0.4 到0.4 就够了
    
    手是 -0.2 到 0.2

    都是背后是正向

"""
from controller import Robot, Motor, Camera, LED, Keyboard, InertialUnit, Gyro, Supervisor, Node, Field
import optparse
import os 
import traceback
import sys
import math
import random
import traceback
import multiprocessing
import time
from multiprocessing.connection import Listener, Connection, PipeConnection, Client
from multiprocessing import shared_memory
class Pedestrian (Supervisor):
    """Control a Pedestrian PROTO."""

    def __init__(self):
        Supervisor.__init__(self)
        """Constructor: initialize constants."""
        self.BODY_PARTS_NUMBER = 13 # 13个组成部分 4*3+1
        self.WALK_SEQUENCES_NUMBER = 8 # 行走的一个周期是8个步长
        self.kick_sequence_number = 10
        self.left_arms_number = 10
        self.ROOT_HEIGHT = 1.27 # 中心的高度
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.speed = 1.15 # 速度
        self.current_height_offset = 0 # 高度补偿
        self.joints_position_field:list[Field] = [] # 
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        
        self.height_offsets = [  # those coefficients are empirical coefficients which result in a realistic walking gait # 经验系数?
            -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03
        ]
        self.angles = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],  # left arm
            [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],  # left lower arm
            [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],  # left hand
            [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],  # right arm
            [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],  # right lower arm
            [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],  # right hand
            [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],  # left leg
            [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],  # left lower leg
            [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],  # left foot
            [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],  # right leg
            [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],  # right lower leg
            [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],  # right foot
            [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]  # head
        ]

        self.kick_angles = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # left arm
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # left lower arm
            [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],  # left hand
            [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],  # right arm
            [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],  # right lower arm
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0],  # right hand
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0],  # left leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0],  # left lower leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0],  # left foot
            
            [0.0, -0.5, -1.0,  -1.7, -1.7,-1.7, -1.7,  -1.2,  -0.9,  -0.3],  # right leg
            [0.0,  0.5,   0.8,  1.2,  1.2, 1.2,  1.2,   0.8,   0.0,    0.0],  # right lower leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0,   0.0,    0.0],            # right foot
            [0.0,  0.0,   0.0,  0.0,  0,0, 0.0, 0.0,   0.0,   0.0,    0.0]  # head
        ]
        """
            0    -0.5   -1.0   -1.7   -1.2   -0.7  0     0
            0     0.5    0.8    1.2    0.8    0    0     0
            0       0    0      0.0     0     0    0     0
        
        """
        self.left_arms = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            

            [-0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5, 0.0, 0.0],  # left arm
            [0.0,  -0.3, -0.6, -0.6, -0.6, -0.6, -0.3,  0.0, 0.0, 0.0],  # left lower arm
            [0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0], # left hand

            [-0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5, 0.0, 0.0],  # left arm
            [0.0,  -0.3, -0.6, -0.6, -0.6, -0.6, -0.3,  0.0, 0.0, 0.0],  # left lower arm
            [0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0], # left hand


            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # left leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # left lower leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # left foot
            
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # right leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # right lower leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # right foot
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]  # head
        ]

        """
        
        """

        # 具体的角度，可以直接拆分节点，去观察， 然后保存下来就ok

        self.time_step = int(self.getBasicTimeStep())
        
        self.speed = 2
        


    def run(self):
        """Set the Pedestrian pose and position."""
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--trajectory", default="", help="Specify the trajectory in the format [x1 y1, x2 y2, ...]")
        opt_parser.add_option("--speed", type=float, default=0.5, help="Specify walking speed in [m/s]") # 这里的输入要注意,webots输入有空格可能会无法识别
        opt_parser.add_option("--step", type=int, help="Specify time step (otherwise world time step is used)")
        options, _ = opt_parser.parse_args() 
        if not options.trajectory or len(options.trajectory.split(',')) < 2:
            print("You should specify the trajectory using the '--trajectory' option.")
            print("The trajectory should have at least 2 points.")
            return
        if options.speed and options.speed > 0:
            self.speed = options.speed
        """         if options.step and options.step > 0:
            self.time_step = options.step
        else:
            self.time_step = int(self.getBasicTimeStep()) """
        point_list = options.trajectory.split(',') # 4 5, 4 -4
        self.number_of_waypoints = len(point_list)
        self.waypoints = [] # 存储路径点
        for i in range(0, self.number_of_waypoints):
            self.waypoints.append([])
            self.waypoints[i].append(float(point_list[i].split()[0])) # 获取路径坐标
            self.waypoints[i].append(float(point_list[i].split()[1]))
        self.root_node_ref = self.getSelf() # 返回Node 节点
        self.root_translation_field = self.root_node_ref.getField("translation") # 获取位置
        self.root_rotation_field = self.root_node_ref.getField("rotation") # 获取旋转??
        for i in range(0, self.BODY_PARTS_NUMBER):
            self.joints_position_field.append(self.root_node_ref.getField(self.joint_names[i])) # 把所有关节的对象存起来

        # compute waypoints distance
        self.waypoints_distance = [] # 计算途径点的距离
        for i in range(0, self.number_of_waypoints):
            x = self.waypoints[i][0] - self.waypoints[(i + 1) % self.number_of_waypoints][0] # 取余还会算最后一个和第0个
            y = self.waypoints[i][1] - self.waypoints[(i + 1) % self.number_of_waypoints][1]
            if i == 0:
                self.waypoints_distance.append(math.sqrt(x * x + y * y))
            else:
                self.waypoints_distance.append(self.waypoints_distance[i - 1] + math.sqrt(x * x + y * y)) # 这个距离是累加起来的


        while not self.step(self.time_step) == -1:
            time = self.getTime() # 每个time_step是0.032 
            
            current_sequence = int(((time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER) # 不管怎么缩放,应该都是 0-WALK_SEQUENCES_NUMBER-1,, 如果放大系数很大 则序列停留短,否则停留长

            # print(current_sequence)
            # compute the ratio 'distance already covered between way-point(X) and way-point(X+1)'
            # / 'total distance between way-point(X) and way-point(X+1)'
            ratio = (time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO - int(((time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO)) # int直接取整,会丢掉所有小数部分,而计算小数可以

            for i in range(0, self.BODY_PARTS_NUMBER): # 遍历所有关节
                current_angle = self.angles[i][current_sequence] * (1 - ratio) + self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio # 让动作可以连续过度到下一个
                self.joints_position_field[i].setSFFloat(current_angle) # 1维的数据

            # adjust height
            self.current_height_offset = self.height_offsets[current_sequence] * (1 - ratio) + self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio # 高度补偿

            # move everything
            distance = time * self.speed

            temp = self.waypoints_distance[self.number_of_waypoints - 1]

            relative_distance = distance - int(distance / temp) * temp # distance < temp 的时候 相对距离就是 distance , distance > temp 的时候 要减去temp

            # 因此 relative 的距离 永远被限制在了 0-temp 之间,

            for i in range(0, self.number_of_waypoints):
                if self.waypoints_distance[i] > relative_distance: # 看哪些点已经过去了????
                    break

            distance_ratio = 0
            if i == 0: # 第0个点还没走
                distance_ratio = relative_distance / self.waypoints_distance[0]
            else:
                distance_ratio = (relative_distance - self.waypoints_distance[i - 1]) / (self.waypoints_distance[i] - self.waypoints_distance[i - 1]) # 在两个点之间走到 什么程度

            x = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][0] + (1 - distance_ratio) * self.waypoints[i][0]
            y = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][1] + (1 - distance_ratio) * self.waypoints[i][1]

            root_translation = [x, y, self.ROOT_HEIGHT + self.current_height_offset]
            
            angle = math.atan2(self.waypoints[(i + 1) % self.number_of_waypoints][1] - self.waypoints[i][1], self.waypoints[(i + 1) % self.number_of_waypoints][0] - self.waypoints[i][0])
            rotation = [0, 0, 1, angle] # 朝着下一个方向转过去 绕z轴旋转

            self.root_translation_field.setSFVec3f(root_translation)

            self.root_rotation_field.setSFRotation(rotation)

myhuman = Pedestrian()

human1 = myhuman.getFromDef('human2')
dog1 = myhuman.getFromDef('dog1')
flag = myhuman.getFromDef('flag')

dog1_trans = dog1.getField('translation')
dog1_rota = dog1.getField('rotation')
dig1_data = dog1.getField('customData')

human1_trans = human1.getField('translation')
human1_rota = human1.getField('rotation')
human1_color = human1.getField('shirtColor')
flag_trans = flag.getField('translation')


init_dx, init_dy, init_dz = dog1.getPosition() # 初始位置，
init_hx, init_hy, init_hz = human1.getPosition()

for i in range(0, myhuman.BODY_PARTS_NUMBER):
    myhuman.joints_position_field.append(human1.getField(myhuman.joint_names[i])) # 把所有关节的对象存起来


def rotate_to_dog(direction=1):
    # 人 向狗的方向转过去
    dx, dy, dz = dog1.getPosition() # 初始位置，
    hx, hy, hz = human1.getPosition()
    angle = math.atan2(dy-hy, dx-hx)
    if direction ==1:
        human1_rota.setSFRotation([0,0,1,angle]) # 转过去
    else:
        human1_rota.setSFRotation([0,0,1,angle+math.pi]) # 转过去
def go_to_dog():
    rotate_to_dog()
    dx, dy, dz = dog1.getPosition() # 初始位置，
    hx, hy, hz = human1.getPosition()
    destination_dis = math.sqrt((dx-hx)**2 + (dy-hy)**2) # 计算终点
    time0 = myhuman.getTime()
    while myhuman.step(myhuman.time_step) != -1:
        
        temp_hx, temp_hy,_ = human1.getPosition()
        temp_dx, temp_dy,_ = dog1.getPosition()

        try:

            time = myhuman.getTime() # 每个time_step是0.032 

            if math.sqrt((temp_hx-temp_dx)**2 + (temp_hy-temp_dy)**2) > 1:
                # 当距离远的时候， 要走过去
                # 生成0-7的序列
                current_sequence = int(((time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO) % myhuman.WALK_SEQUENCES_NUMBER) # 不管怎么缩放,应该都是 0-WALK_SEQUENCES_NUMBER-1,, 如果放大系数很大 则序列停留短,否则停留长
                # 找到当时的比例
                ratio = (time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO - int(((time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO)) # int直接取整,会丢掉所有小数部分,而计算小数可以

                for i in range(0, myhuman.BODY_PARTS_NUMBER): # 遍历所有关节，关节移动
                    current_angle = myhuman.angles[i][current_sequence] * (1 - ratio) + myhuman.angles[i][(current_sequence + 1) % myhuman.WALK_SEQUENCES_NUMBER] * ratio # 让动作可以连续过度到下一个
                    myhuman.joints_position_field[i].setSFFloat(current_angle) # 1维的数据

                # move everything
                distance = (time-time0) * myhuman.speed

                distance_ratio = distance / destination_dis # 计算位置 比例

                x = distance_ratio * dx + (1 - distance_ratio) * hx
                y = distance_ratio * dy + (1 - distance_ratio) * hy

                human1_trans.setSFVec3f([x, y, hz])
            else:
                for i in range(0, myhuman.BODY_PARTS_NUMBER): # 遍历所有关节，关节移动
                    myhuman.joints_position_field[i].setSFFloat(0.0) # 立正站好
                break


        except:
            a, b, c = sys.exc_info()
            print(traceback.format_exception(a,b,c))
def go_away_dog():

    rotate_to_dog(-1)

    dx, dy, dz = dog1.getPosition() # 初始位置，
    hx, hy, hz = human1.getPosition()
    time0 = myhuman.getTime()
    new_hx = (hx-dx)*3 +hx
    new_hy = (hy-dy)*3 +hy

    destination_dis = math.sqrt((dx-new_hx)**2 + (dy-new_hy)**2) # 计算终点
    while myhuman.step(myhuman.time_step) != -1:
        
        temp_hx, temp_hy,_ = human1.getPosition()
        temp_dx, temp_dy,_ = dog1.getPosition()

        try:

            time = myhuman.getTime() # 每个time_step是0.032 

            if math.sqrt((temp_hx-temp_dx)**2 + (temp_hy-temp_dy)**2) <10:
                # 当距离远的时候， 要走过去
                # 生成0-7的序列
                current_sequence = int(((time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO) % myhuman.WALK_SEQUENCES_NUMBER) # 不管怎么缩放,应该都是 0-WALK_SEQUENCES_NUMBER-1,, 如果放大系数很大 则序列停留短,否则停留长
                # 找到当时的比例
                ratio = (time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO - int(((time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO)) # int直接取整,会丢掉所有小数部分,而计算小数可以

                for i in range(0, myhuman.BODY_PARTS_NUMBER): # 遍历所有关节，关节移动
                    current_angle = myhuman.angles[i][current_sequence] * (1 - ratio) + myhuman.angles[i][(current_sequence + 1) % myhuman.WALK_SEQUENCES_NUMBER] * ratio # 让动作可以连续过度到下一个
                    myhuman.joints_position_field[i].setSFFloat(current_angle) # 1维的数据

                # move everything
                distance = (time-time0) * myhuman.speed


                distance_ratio = distance / destination_dis # 计算位置 比例
                # print(distance_ratio)
                x = distance_ratio * new_hx + (1 - distance_ratio) * hx
                y = distance_ratio * new_hy + (1 - distance_ratio) * hy

                human1_trans.setSFVec3f([x, y, hz])
            else:
                for i in range(0, myhuman.BODY_PARTS_NUMBER): # 遍历所有关节，关节移动
                    myhuman.joints_position_field[i].setSFFloat(0.0) # 立正站好
                break


        except:
            a, b, c = sys.exc_info()
            print(traceback.format_exception(a,b,c))
def stand_up():
    for i in range(0, myhuman.BODY_PARTS_NUMBER): # 遍历所有关节，关节移动
        myhuman.joints_position_field[i].setSFFloat(0.0) # 立正站好
def wait_seconds(n):
    i = 1
    while myhuman.step(myhuman.time_step) != -1 and i<=n:
        i+=1
        time.sleep(0.1)
def kick(dx, dy, hx, hy, force=1):

    time0 = myhuman.getTime()
    while myhuman.step(myhuman.time_step) != -1:
        temp_time = myhuman.getTime() - time0
        current_sequence = int(((temp_time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO) % myhuman.kick_sequence_number)

        ratio = (temp_time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO - int(((temp_time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO))
        for i in range(0, myhuman.BODY_PARTS_NUMBER): # 遍历所有关节，关节移动
            current_angle = myhuman.kick_angles[i][current_sequence] * (1 - ratio) + myhuman.kick_angles[i][(current_sequence + 1) % myhuman.kick_sequence_number] * ratio # 让动作可以连续过度到下一个
            myhuman.joints_position_field[i].setSFFloat(current_angle) # 1维的数据
        if current_sequence == myhuman.kick_sequence_number -1:
            dog1.addForce([(dx-hx)*100 * force, (dy-hy)*100*force, 0], True)
            stand_up()
            break
def hands_up():
    time0 = myhuman.getTime()
    while myhuman.step(myhuman.time_step) != -1:
        temp_time = myhuman.getTime() - time0
        current_sequence = int(((temp_time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO) % myhuman.left_arms_number)

        ratio = (temp_time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO - int(((temp_time * myhuman.speed) / myhuman.CYCLE_TO_DISTANCE_RATIO))
        for i in range(0, myhuman.BODY_PARTS_NUMBER): # 遍历所有关节，关节移动
            current_angle = myhuman.left_arms[i][current_sequence] * (1 - ratio) + myhuman.left_arms[i][(current_sequence + 1) % myhuman.left_arms_number] * ratio # 让动作可以连续过度到下一个
            myhuman.joints_position_field[i].setSFFloat(current_angle) # 1维的数据
        if current_sequence == myhuman.left_arms_number -1:
            stand_up()
            break

def stand_behind_dog():
    dx, dy, dz = dog1.getPosition() # 初始位置
    ls = dog1.getOrientation() # 0 3 分别代表狗这时候的朝向的xy 向量
    t_x = ls[0]
    t_y = ls[3] # 可以对这个向量进行0-PI/3的旋转
    rand_theta = random.uniform(-math.pi/3,math.pi/3) # 也就是出现在 狗的 后面 扇形-pi/3 ,pi/3的区间里
    costheta = math.cos(rand_theta)
    sintheta = math.sin(rand_theta)
    t_x_new = costheta * t_x - sintheta * t_y # 旋转后的向量
    t_y_new = sintheta * t_x + costheta * t_y
    hx, hy, hz = human1.getPosition()

    new_hx = dx - t_x_new*10
    new_hy = dy - t_y_new*10

    human1_trans.setSFVec3f([new_hx, new_hy, hz])   
    myhuman.step(myhuman.time_step)

def go_back_and_kick():
        # 走到后面去给狗狠狠来一下
        if dig1_data.getSFString() == 'test':
            wait_time = 10
        else:
            wait_time = 10
        stand_behind_dog()
        dx, dy, dz = dog1.getPosition() # 初始位置，决定了力的方向
        hx, hy, hz = human1.getPosition()
        go_to_dog()
        stand_up()
        wait_seconds(wait_time)
        if dig1_data.getSFString() != 'test' or dig1_data.getSFString() == 'micc': # 测试的时候不踢, micc的时候还是要踢
            kick(dx, dy, hx, hy,1.5)
        wait_seconds(wait_time)
        go_away_dog()
        stand_up()
        wait_seconds(wait_time)


# temp_time = time.strftime('%d-%H-%M',time.localtime(time.time()))


if __name__ == '__main__':

    while True:
        if flag.getPosition()[1] < 50: # 训练过程
            if flag.getPosition()[0] > 50:
                """ co = myhuman.getSelf().getField('shirtColor')
                co.setSFVec3f([0,1,1]) """  # 设置颜色总会让控制器重新跑一次，不知道为什么
                # print("dog oritation", dog1.getOrientation())
                go_back_and_kick()
                flag_trans.setSFVec3f([0,0,0])
                myhuman.step(myhuman.time_step)
            else:
                myhuman.step(myhuman.time_step)
        else: # 加入小紫
            if 90 < flag.getPosition()[0] < 110:
                """ co = myhuman.getSelf().getField('shirtColor')
                co.setSFVec3f([0,1,1]) """  # 设置颜色总会让控制器重新跑一次，不知道为什么
                # print("dog oritation", dog1.getOrientation())
                go_back_and_kick()
                flag_trans.setSFVec3f([150,100,0])
                myhuman.step(myhuman.time_step)
            else:
                myhuman.step(myhuman.time_step)
        
    