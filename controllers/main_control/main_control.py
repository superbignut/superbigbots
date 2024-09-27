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
import traceback
import sys

# print(sys.version) # 3.9.13
# print(sys.executable) # pytorch

human1 = mydog.getFromDef("human1") # 获取全局对象，进而方便获得位置和速度信息

human2 = mydog.getFromDef("human2")

dog1 = mydog.getFromDef("dog1") # 获取全局对象

flag1 = mydog.getFromDef('flag')

myenv = Env(dim=10, human=human1, dog=dog1, flag=flag1, human1=human1, human2=human2) # 初始化相对速度和位置, 这个作为环境输入，这里也会启动状态检测线程，dim=多少个时间的数据

mybot = Cobot(webots_robot=mydog, env=myenv) # 二阶段下发robot

is_train = True

while mydog.step(timestep) != -1:
    try:
        # state_input = myenv.get_input() # 获取输入, 这里如果可以，控制获取数据的频率就好了， 不要获取的太快
        if is_train:
            mybot.train()
        else:
            mybot.step() # 把输入传给robot, 具体的动作相应也放在robot的函数中实现
    except:
        print("some_error occured")
        a,b,c = sys.exc_info()
        print(traceback.format_exception(a,b,c))
        continue
    time.sleep(0.01)