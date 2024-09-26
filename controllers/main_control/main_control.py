from controller import Robot, Motor, Camera, LED, Keyboard, InertialUnit, Gyro, Supervisor, Node, Field
from action_define import robot as mydog
from device_init import *
from action_define import *
from collections import deque
import numpy as np
from cobot import Cobot, Env
import math
import time
from SPAIC import spaic
import torch
import multiprocessing
import os
import collections
import random
import threading
import sys

# print(sys.version) # 3.9.13
# print(sys.executable) # pytorch

human1 = mydog.getFromDef("human1") # 获取全局对象，进而方便获得位置和速度信息

dog1 = mydog.getFromDef("dog1") # 获取全局对象

flag1 = mydog.getFromDef('flag')

mybot = Cobot(webots_robot=mydog) # 二阶段下发robot

myenv = Env(dim=10, human=human1, dog=dog1, flag=flag1) # 初始化相对速度和位置, 这个作为环境输入，这里也会启动状态检测线程，dim=多少个时间的数据

while mydog.step(timestep) != -1:
    try:
        state_input = myenv.get_input() # 获取输入, 这里如果可以，控制获取数据的频率就好了， 不要获取的太快
        mybot.step(state=state_input) # 把输入传给robot, 具体的动作相应也放在robot的函数中实现

    except:
        print("some_error occured")
        continue
    time.sleep(0.01)