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

human3 = mydog.getFromDef("human3")

dog1 = mydog.getFromDef("dog1") # 获取全局对象

flag1 = mydog.getFromDef('flag')

myenv = Env(dim=10, human=human1, dog=dog1, flag=flag1, human1=human1, human2=human2, human3=human3) # 初始化相对速度和位置, 这个作为环境输入，这里也会启动状态检测线程，dim=多少个时间的数据

mybot = Cobot(webots_robot=mydog, env=myenv) # 二阶段下发robot

is_train = 4 # 1 train 2 test 3 mic_change 4 after_micc_test

if is_train == 1:
    while True:
        try:
            # state_input = myenv.get_input() # 获取输入, 这里如果可以，控制获取数据的频率就好了， 不要获取的太快
            mybot.train()
        except:
            print("some_error occured")
            a,b,c = sys.exc_info()
            print(traceback.format_exception(a,b,c))
            continue
        time.sleep(0.01)       

elif is_train == 2: # 两个小人的测试
    cdata = dog1.getField('customData')
    cdata.setSFString("test") # 用于控制 human controllor 在测试的时候不用发出动作
    mybot.load_model_and_buffer() # 加载一阶段训练数据
    mybot.test()
    while True:
        try:            
            pass
        except:
            print("some_error occured")
            a,b,c = sys.exc_info()
            print(traceback.format_exception(a,b,c))
            continue
        time.sleep(0.01)  
elif is_train == 3: # 三个小人的测试， 需要手动 吧flag1坐标改为 0，100， 0
    """ flag_trans = flag1.getField('translation')
    flag_trans.setSFVec3f([0,100,0])   """
    cdata = dog1.getField('customData')
    cdata.setSFString("test") # 用于控制 human controllor 在测试的时候不用发出动作
    mybot.load_model_and_buffer() # 加载一阶段训练数据
    mybot.test()
    while True:
        try:            
            pass
        except:
            print("some_error occured")
            a,b,c = sys.exc_info()
            print(traceback.format_exception(a,b,c))
            continue
        time.sleep(0.01)       
elif is_train == 4: # 三个小人进行微调训练
    cdata = dog1.getField('customData')
    cdata.setSFString("micc")
    mybot.load_model_and_buffer() # 
    
    # mybot.test()
    # while mydog.step(timestep) != -1:
    while True:
        try:
            mybot.mic_change_and_train()
        except:
            print("some_error occured")
            a,b,c = sys.exc_info()
            print(traceback.format_exception(a,b,c))
            continue
        time.sleep(0.01)      
""" else:
    # 这个是eval 之后的test， 可以分别测试一下，他们之间的准确率的区别
    cdata = dog1.getField('customData')
    cdata.setSFString("test")
    mybot.load_model_and_buffer_after_micc() # 加载训练之后的模型
    #mybot.load_model_and_buffer() # 加载训练之前的模型
    mybot.test_for_plot()
    while mydog.step(timestep) != -1:
        try:            
            pass
        except:
            print("some_error occured")
            a,b,c = sys.exc_info()
            print(traceback.format_exception(a,b,c))
            continue
        time.sleep(0.01)      
         """