from controller import Robot, Motor, Camera, LED, Keyboard, InertialUnit, Gyro, Node, Supervisor
from device_init import *
from action_define import *
from collections import deque
from action_define import robot as mydog
import numpy as np
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

USER_CMD = ["sit_down", "lie_down", "go_head", "go_back", "give_paw", "stand_up", "null"]

USER_CMD_DICT = {"sit_down":0, "lie_down":1, "go_head":2, "go_back":3, "give_paw":4, "stand_up":5, "null":6}

state_nums = 10

# human1 = mydog.getFromDef("human1") # 得到human1的handler

class Env:
    # 类似rl中的环境类，调用get_input可以获取到环境信息
    # 目前是 10 * 6 = 相位位置*2、速度*2、指令*1、振动输入*1
    def __init__(self, dim, human:Node, dog:Node, rela=True) -> None:
        self.rela = rela
        self.human = human
        self.dog = dog

        self.user_cmd = None # 检测到用户输入命令 0-len(USER_CMD_DICT)
        self.gyro_err = 0 # 检测到 陀螺仪异常 0 1

        self.user_cmd_thread = threading.Thread(target=self._get_user_cmd, name="_get_user_cmd") # 线程中不断修改用户当下的指令，只保存最近的这个指令
        self.gyro_err_thread = threading.Thread(target=self._get_gyro, name="_get_gyro") # gyro 检测异常振动，这里是在模拟，被踢的时候

        # if self.rela == True:
        assert self.rela is True, "rela must be true"

        self.human_position = deque([np.array(human.getPosition()[0:2]) - np.array(dog.getPosition()[0:2])  for _ in range(dim)], maxlen=dim) # 相对位置初始化

        self.human_speed = deque([np.array([0,0]) for _ in range(dim)], maxlen=dim) # 相对速度初始化

        self.human_cmd = deque([[USER_CMD_DICT['null']] for _ in range(dim)], maxlen=dim) # 指令序列初始化

        self.dog_gyro = deque([[0] for _ in range(dim)], maxlen=dim) # 指令序列初始化

        self._all_thread_start()


        
    def _all_thread_start(self):
        self.user_cmd_thread.start()
        self.gyro_err_thread.start()


    def get_input(self):
        # 每次更新一次queue 尾部的数据， 并去除队头的数据
        self.human_position.append(np.array(self.human.getPosition()[0:2]) - np.array(self.dog.getPosition()[0:2])) # 10 * 2 
        
        self.human_speed.append((self.human_position[-1] - self.human_position[-2])) # 这里直接用差值作为相对速度了， 暂时 没有考虑狗的速度 # 10 * 2

        self.human_cmd.append([self.user_cmd]) # 10 * 1

        self.dog_gyro.append([self.gyro_err]) # 10 * 1

        return np.concatenate((self.human_position, self.human_speed, self.human_cmd, self.dog_gyro), axis=1) # 暂时的数据维度是10 * 6 但是没有进行归一化
    

    def _get_user_cmd(self):
        # 每个除 null 之外的指令，被检测到后都会延迟一段时间再进行下一次检测
        while True:
            key = keys.getKey()

            if key == ord('A'):
                self.user_cmd = USER_CMD_DICT["sit_down"]
                time.sleep(0.5) 
            elif key == ord('B'):
                self.user_cmd = USER_CMD_DICT["give_paw"]
                time.sleep(0.5)
            else:
                self.user_cmd = USER_CMD_DICT["null"]
        
    def _get_gyro(self):
        while True:
            _gyro_values =gyro.getValues()            
            if(max(_gyro_values)) > 1.5:
                self.gyro_err = 1
                time.sleep(0.5)
                self.gyro_err = 0

class REMONET:
    def __init__(self) -> None:
        # 两个网络的定义都在这里
        pass

    def get_emotion(self, state):
        # 从emo网络 获得emo输出
        return 0
    
    def get_reward(self):
        # 从reward 网络获得 reward输出
        pass
    
    def update(self):
        # 根据交互结果进行更新reward
        pass

class Cobot:
    # 二阶段机器人

    def __init__(self, webots_robot) -> None:
        self.webots_robot = webots_robot # 仿真机器人对象
        self.user_cmd = None # 
        self.emo_model = REMONET()

    def step(self, state):
        # 读取输入
        # 检测指令
        # 情感模型生成情感
        # 生成二阶动作
        # --- 等待交互结束 ---
        # 交互结果更rstdp新情感模型
        if self.user_cmd == USER_CMD_DICT["null"]:
            # 没有指令的时候 趴下
            lie_down(1.0) #
        else:
            # 一阶段的动作
            self._cmd_to_action_1(self.user_cmd) # 启动一个线程
            # 情感模型触发， 这里可以和一阶段同时运行，否则还要进行等待，
            temp_emo = self.emo_model.get_emotion(state)

            # 二阶段的动作
            self._cmd_to_action_2(self.user_cmd, temp_emo)

            self.wait_interact() # 

    def wait_interact(self):
        # 等待交互结果
        # 这里启动一个监听线程 就ok
        self.emo_model.update()


    def _cmd_to_action_2(self, cmd, emo):

        pass


    def _cmd_to_action_1(self, cmd):
        # 一阶段，执行相应的动作, 都是坐下
        if cmd == USER_CMD_DICT["sit_down"]: # ["sit_down", "lie_down", "go_head", "go_back", "give_paw", "stand_up", "null"]
            sit_down(1.0)
        elif cmd == USER_CMD_DICT["lie_down"]:
            sit_down(1.0)
        elif cmd == USER_CMD_DICT["go_head"]:
            sit_down(1.0)
        elif cmd == USER_CMD_DICT["go_back"]:
            sit_down(1.0)
        elif cmd == USER_CMD_DICT["give_paw"]:
            sit_down(1.0)
        elif cmd == USER_CMD_DICT["stand_up"]:
            sit_down(1.0)

        


    def get_output_action(self):
        # 输出动作-二阶段动作
        pass

    def _get_inner_action(self):
        # 不输出动作-一阶段动作
        pass

    def _get_inner_emotion(self):
        # 获取情感状态
        pass

    def _get_reward_signal(self):
        # 对于一次 get_output_action 如果交互成功 或者 不交互成功，就 正向强化，否则弱化
        pass

    