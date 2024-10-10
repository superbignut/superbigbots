from controller import Robot, Motor, Camera, LED, Keyboard, InertialUnit, Gyro, Node, Supervisor
from device_init import *
from action_define import *
from collections import deque
from action_define import robot as mydog
from cobot_emo_net import EMONET, model_path, buffer_path, device, mic_buffer_path, mic_model_path
import numpy as np
import math
import multiprocessing
import time
from multiprocessing.connection import Listener, Connection, PipeConnection, Client

from SPAIC import spaic
import torch
import os
import collections
import random
import threading
import sys
from threading import Lock

from queue import Queue

from multiprocessing import shared_memory

USER_CMD = ["sit_down", "lie_down", "go_head", "go_back", "give_paw", "stand_up", "null"]

USER_CMD_DICT = {"sit_down":0, "lie_down":1, "go_head":2, "go_back":3, "give_paw":4, "stand_up":5, "null":6}

state_nums = 10

# human1 = mydog.getFromDef("human1") # 得到human1的handler

class Env:
    # 类似rl中的环境类，调用get_input可以获取到环境信息
    # 目前是 10 * 6 = 相位位置*2、速度*2、指令*1、振动输入*1
    def __init__(self, dim, human:Node, dog:Node,flag:Node=None,rela=True,human1:Node=None, human2:Node=None, human3:Node=None) -> None:
        self.rela = rela
        self.human = human
        self.dog = dog
        self.flag=flag
        self.flag_cmd_data = self.flag.getField('name') # name用来做指令的flag
        self.dim = dim # 这里的dim 指的是 过去多少个时间片的数据
        self.human1 = human1
        self.human2 = human2
        self.human3 = human3
        self.snn_distance = 40

        self.user_cmd = USER_CMD_DICT['null'] # 检测到用户输入命令 0-len(USER_CMD_DICT)
        self.user_cmd_lock = Lock() # 用于 场景中 人的动作执行时会直接修改 dog 的  cmd参数， 因此可能会去检测线程冲突，所以加锁
        # 不同文件之间通信可能要使用 网络 或者文件来做
        self.gyro_err = 0 # 检测到 陀螺仪异常 0 1

        self.user_cmd_thread = threading.Thread(target=self._get_user_cmd, name="_get_user_cmd") # 线程中不断修改用户当下的指令，只保存最近的这个指令
        self.gyro_err_thread = threading.Thread(target=self._get_gyro, name="_get_gyro") # gyro 检测异常振动，这里是在模拟，被踢的时候
        self.listen_thread = threading.Thread(target=self._listen_cmd, name='_listen_cmd') # 进程间通信
        self.xy_pos_limit = 20 # 归一化常数
        self.xy_speed_limit = 4 # 根据实际修改 # 这个速度可能要加快一点，现在的人的移动速度有点太慢了
        self.cmd_limit = len(USER_CMD_DICT)-1 # 指令归一化

        # if self.rela == True:
        assert self.rela is True, "rela must be true"

        self.human_position = deque([ np.clip(np.array(human.getPosition()[0:2]) - np.array(dog.getPosition()[0:2]),-self.xy_pos_limit,self.xy_pos_limit)/self.xy_pos_limit  for _ in range(dim)], maxlen=dim) # 相对位置初始化

        self.human_speed = deque([np.array([0,0]) for _ in range(dim)], maxlen=dim) # 相对速度初始化

        self.human_cmd = deque([[USER_CMD_DICT['null'] / self.cmd_limit] for _ in range(dim)], maxlen=dim) # 指令序列初始化

        self.dog_gyro = deque([[0] for _ in range(dim)], maxlen=dim) # 指令序列初始化


        self.human_color = 0
        self.human_speed = 0
        self.human_path_direction = np.array([0,0]) # 与dog的正前方 相对的角度 只考虑xy平面

        


        self._all_thread_start()
        
    def _listen_cmd(self):
        """ temp_time = time.strftime('%d-%H-%M',time.localtime(time.time()))
        shm_a = shared_memory.SharedMemory(name = 'cmdcmd' + temp_time, create=True, size=10) # 创建一个共享内存 """

        while True: # 

            if self.flag.getField('name').getSFString() == 'standup':
                self.set_cmd_with_lock(USER_CMD_DICT["stand_up"])
                time.sleep(0.5)
                # print('receive standup cmd')
                self.flag.getField('name').setSFString('null') # 这里的修改逻辑可能有问题
            
            self.set_cmd_with_lock(USER_CMD_DICT["null"])

        
    def _all_thread_start(self):
        # self.user_cmd_thread.start()
        self.gyro_err_thread.start()
        self.listen_thread.start()


    def get_input(self):
        # 每次更新一次queue 尾部的数据， 并去除队头的数据, 返回的是一个 shape = (10, 6) 的np.array
        self.human_position.append(np.clip(np.array(self.human.getPosition()[0:2]) - np.array(self.dog.getPosition()[0:2]), -self.xy_pos_limit, self.xy_pos_limit)/self.xy_pos_limit ) # 10 * 2 
        
        self.human_speed.append(np.clip(self.human_position[-1] - self.human_position[-2], -self.xy_speed_limit, self.xy_speed_limit) / self.xy_speed_limit ) # 这里直接用差值作为相对速度了， 暂时 没有考虑狗的速度 # 10 * 2

        self.human_cmd.append([self.user_cmd / self.cmd_limit]) # 10 * 1

        self.dog_gyro.append([self.gyro_err]) # 10 * 1

        return np.concatenate((self.human_position, self.human_speed, self.human_cmd, self.dog_gyro), axis=1) # 暂时的数据维度是10 * 6 但是没有进行归一化
    
    def get_three_human_input(self):
        # 这里应该只是要离dog 近的那个人就可以了，所以还需要一个判断吗
        # 数据做的简单一点
        # 一个是衣服颜色， 一个是过来的路线的方向 # 还可以加上速度 
        # 1 + 2 + 1 现在是 4个输入， 现在的目标就是把这四个输入训练出来

        temp_return = np.zeros(12) # 前两个是颜色和速度， 后8 个是四个象限的速度 ++ +- -- -+
        temp_human = self.which_human_near()
        # print(temp_human)
        if temp_human == 1:
            color = 0 # green
            speed = 0 # 这里默认给绿色小人的速度是1
            rela_pos = np.array(self.human1.getPosition()[0:2]) - np.array(self.dog.getPosition()[0:2])
            rela_pos = rela_pos / math.sqrt(rela_pos[0]**2 + rela_pos[1] ** 2) # 方向归一化
            # print(rela_pos)
            temp_return[0] = 1 # 这两个是颜色 绿
            temp_return[1] = 0 # 红

            temp_return[10] = 1 # 这两个是速度 慢
            temp_return[11]= 0 # 快
        elif temp_human == 2:
            color = 1 # red
            speed = 1 # 这里默认给红色小人的速度是 正常的2倍
            rela_pos = np.array(self.human2.getPosition()[0:2]) - np.array(self.dog.getPosition()[0:2])
            rela_pos = rela_pos / math.sqrt(rela_pos[0]**2 + rela_pos[1] ** 2) # 方向归一化

            temp_return[0] = 0 # 这两个是颜色 绿
            temp_return[1] = 1 # 红

            temp_return[10] = 0 # 这两个是速度 慢  但其实颜色和速度 也用分类的这么准确吗，就是用浮点是我感觉也没问题吧
            temp_return[11]= 1 # 快 

            # 如果有更多的速度分类， 或者颜色分类，都可以进行拓展
            # print(rela_pos)
        # return [color, speed, (rela_pos[0] +1) *0.5, (rela_pos[1]+1)*0.5] # 这里我感觉把负数单独用两个维度来表示比较好
        # 衣服颜色和速度，也用不同维度分开
        elif temp_human == 3:
            # print("yes")
            # 这里使用来给紫色 微调小人准备的
            rela_pos = np.array(self.human3.getPosition()[0:2]) - np.array(self.dog.getPosition()[0:2]) # 紫色仍然使用 huaman1 控制器
            rela_pos = rela_pos / math.sqrt(rela_pos[0]**2 + rela_pos[1] ** 2) # 方向归一化
            # print(rela_pos)
            temp_return[0] = 1 # 紫色
            temp_return[1] = 1 

            temp_return[10] = 1 # 这两个是速度 慢
            temp_return[11]= 1 # 快

        if rela_pos[0] >=0 and rela_pos[1] >= 0: # ++
            temp_return[2] = abs(rela_pos[0])
            temp_return[3]= abs(rela_pos[1])
        elif rela_pos[0] >=0 and rela_pos[1] <= 0: # +-
            temp_return[4] = abs(rela_pos[0])
            temp_return[5]= abs(rela_pos[1])
        elif rela_pos[0] <=0 and rela_pos[1] <= 0: # --
            temp_return[6] = abs(rela_pos[0])
            temp_return[7]= abs(rela_pos[1])
        else: # -+
            temp_return[8] = abs(rela_pos[0])
            temp_return[9]= abs(rela_pos[1])
        
        return np.array([temp_return]) # 加了一层batch
    def people_is_near(self):
        # 选择最近的那个人触发 
        
        h1x, h1y = self.human1.getPosition()[0:2]
        h2x, h2y = self.human2.getPosition()[0:2]
        h3x, h3y = self.human3.getPosition()[0:2]
        dx, dy = self.dog.getPosition()[0:2]
        dis1 = (h1x-dx) **2 + (h1y-dy)**2
        dis2 = (h2x-dx) **2 + (h2y-dy)**2
        dis3 = (h3x-dx) **2 + (h3y-dy)**2

        if dis1 < self.snn_distance or dis2  < self.snn_distance or dis3 < self.snn_distance:
            return True
        else:
            return False
        

    def which_human_near(self):
        h1x = self.human1.getPosition()[0]
        h1y = self.human1.getPosition()[1]
        h2x = self.human2.getPosition()[0]
        h2y = self.human2.getPosition()[1]
        h3x = self.human3.getPosition()[0]
        h3y = self.human3.getPosition()[1]

        dx = self.dog.getPosition()[0]
        dy = self.dog.getPosition()[1]

        dis1 = (h1x-dx) **2 + (h1y-dy)**2
        dis2 = (h2x-dx) **2 + (h2y-dy)**2
        dis3 = (h3x-dx) **2 + (h3y-dy)**2        

        if dis1 < dis2:
            if dis1 < dis3:
                return 1
            else:
                return 3
        else:
            if dis2 < dis3:
                return 2
            else:
                return 3
    
        
    def set_cmd_with_lock(self, cmd):
        # lock没有使用
        with self.user_cmd_lock:
            self.user_cmd = cmd

    def _get_user_cmd(self):
        # 每个除 null 之外的指令，被检测到后都会延迟一段时间再进行下一次检测
        while True:
            key = keys.getKey()

            if key == ord('Q'):
                self.set_cmd_with_lock(USER_CMD_DICT["sit_down"])
                time.sleep(0.5)  #  这里的sleep 时间可以设置的和 time_step 有些关系
            elif key == ord('W'):
                self.set_cmd_with_lock(USER_CMD_DICT["give_paw"])
                time.sleep(0.5)
            elif key == ord('E'):
                self.set_cmd_with_lock(USER_CMD_DICT["stand_up"])
                time.sleep(0.5)
            else:
                self.set_cmd_with_lock(USER_CMD_DICT["null"])
        
    def _get_gyro(self):
        time.sleep(8) # 最开始不检查
        while True:
            _gyro_values =gyro.getValues()     
            # print(_gyro_values)       
            if(max(_gyro_values)) > 0.2:
                # print("receive a kick!")
                self.gyro_err = 1
                time.sleep(0.5)
                self.gyro_err = 0



class Cobot:
    # 二阶段机器人

    def __init__(self, webots_robot, env:Env) -> None:
        self.webots_robot = webots_robot # 仿真机器人对象
        # self.user_cmd = USER_CMD_DICT["null"] # 
        self.emo_model = EMONET() # 用于情感输出的spaic在线学习投票网络
        self.env = env
        self.all_sit_down_flag = True


        self.train_i = 0 # 
        self.train_num = 400 # 控制训练次数

        self.micc_i = 0
        self.micc_num = 200 # 微调次数

        self.test_i = 0
        self.right_i = 0
        self.test_num = 200
        self.test_acc_list = []

        self.right_predict = 0
        self.right_predict_list = []

        self.emo_queue = deque(maxlen=3) # 只保存最近的情感预测， 用于比较
    def step(self, state=None):
        # 读取输入
        # 检测指令
        # 情感模型生成情感
        # 生成二阶动作
        # --- 等待交互结束 ---
        # 交互结果更rstdp新情感模型
        # print(state, state.shape)
        if self.env.user_cmd == USER_CMD_DICT["null"]:
            # 没有指令的时候 趴下
            sit_down(1.0) #
        else:
            # 一阶段的动作
            self._cmd_to_action_1(self.user_cmd) # 启动一个线程
            # 情感模型触发， 这里可以和一阶段同时运行，否则还要进行等待，
            temp_emo = self.emo_model.get_emotion(state) # 这里是把所有数据都数据都输入进去呢，还是 有选择的输入呢

            # 二阶段的动作
            self._cmd_to_action_2(self.user_cmd, temp_emo)

            self.wait_interact() # 

    def train(self, state=None):
        # 对网路进行训练
        if self.all_sit_down_flag:
            self.sit_down_all_the_time()
            self.all_sit_down_flag = False


        state = self.env.get_three_human_input() #  数据更细与获取， 这里的输入有负数，需要处理
        if self.train_i > self.train_num:
            return 
        elif self.train_i == self.train_num:
            self.emo_model.pre_train_over_save() # 训练结束保存数据 
            self.train_i += 1
        # else:
            
        temp_label = 2
        if self.env.user_cmd == USER_CMD_DICT["stand_up"]: # 积极label    
            print("positive label")
            temp_label = 0
        elif self.env.gyro_err == 1: # 消极lebel
            print("negetive label")
            temp_label = 1
        if temp_label == 0 or temp_label ==1:
            # print("Net's input data is :",np.array([state]))
            output = self.emo_model.pre_train(data=state, label=temp_label) # positive label # 打印输出看看
            # 这个输出现在是有问题的， 猜测是编码上的输入不对
            print("output is :", output)
            self.train_i += 1 
            # print("Train num is : ", self.train_i)
    
    
    def mic_change_and_train(self):
        # 这里加入的小紫， 要对模型进行微调
        if self.all_sit_down_flag:
            self.sit_down_all_the_time() # 这里就不能一直坐着了，需要根据 情感来进行动作
            self.all_sit_down_flag = False
        
        state = self.env.get_three_human_input() #  数据更细与获取， 这里的输入有负数，需要处理
        if self.micc_i > self.micc_num:
            return 
        elif self.micc_i == self.micc_num:
            pass
            # 暂不保存模型和buffer， 等调好了的
            """ np.savetxt('./cobot_eval_acc.txt', fmt='%f', delimiter=',',X=self.right_predict_list) # 保存预测成功率曲线
            # 这里最后最好还是可以 单独看一下对 小紫的预测情况  
            print(" 总数为：", self.micc_num+2 ," 预测成功为：", self.right_predict)
            self.emo_model.mic_change_over_save() # 微调训练结束保存数据 
            self.micc_i += 1 """
        
        temp_label = 2
        # print(self.env.user_cmd)
        if self.env.user_cmd == USER_CMD_DICT["stand_up"]: # 积极label    
            #print("positive label")
            temp_label = 0
        elif self.env.gyro_err == 1: # 消极lebel
            #print("negetive label")
            temp_label = 1
        

        if temp_label == 0 or temp_label ==1: # 有明确的指令输入时， 这时， 和之前预测的多个情感进行比较
            # print("Net's input data is :",np.array([state]))
            output = self.emo_model.step(data=state) # positive label # 打印输出看看
            
            predict_label = self.check_emo_queue()

            self.emo_model.assign_label_update(newoutput=output, newlabel=temp_label) # 更新数据

            if predict_label == temp_label:
                self.right_predict += 1
            print("the predict_label is: ", predict_label, " the right_label is: ", temp_label)
            
            self.right_predict_list.append(self.right_predict / (self.micc_i+1)) # 把实时的预测成功占比 写在 list中
            self.micc_i += 1 
        else: # 这里是只有人接近机器狗的时候， 机器狗也会有一个预测，区别于交互， 这里并不修改buffer 和model
            output = self.emo_model.step(data=state) # positive label # 打印输出看
            temp_cnt = [0 for _ in range(len(self.emo_model.buffer))] # 维度是2
            for i in range(len(self.emo_model.assign_label)):
                temp_cnt[self.emo_model.assign_label[i]] += output[0, i] # 第一个维度是batch, 
            predict_label = torch.argmax(torch.tensor(temp_cnt))
            self.emo_queue.append(predict_label) # 将最近的预测结果放入队列
            print("the predict_label is: ", self.check_emo_queue())


    def check_emo_queue(self):
        if self.emo_queue.count(0) > self.emo_queue.count(1): # 看谁的数量多
            return 0
        else:
            return 1
        
    def load_model_and_buffer(self):
        self.emo_model.state_from_dict(filename=model_path, device=torch.device("cuda"))
        self.emo_model.buffer = torch.load(buffer_path)
        self.emo_model.assign_label_update() # 更新buffer

    def load_model_and_buffer_after_micc(self):
        self.emo_model.state_from_dict(filename=mic_model_path, device=torch.device("cuda"))
        self.emo_model.buffer = torch.load(mic_buffer_path)
        self.emo_model.assign_label_update() # 更新buffer

    def test(self, state=None):
        # 调用之前需要先加载模型参数和buffer -> load_model_and_buffer
        if self.all_sit_down_flag:
            self.sit_down_all_the_time()
            self.all_sit_down_flag = False

        test_thread = threading.Thread(target=self._test_thread, name="test_thread")
        test_thread.start()
        
            # time.sleep(0.5)
    def _test_thread(self):
        while True:
            state = self.env.get_three_human_input()
            
            with torch.no_grad():
                # print(state)
                output = self.emo_model.step(state)

                temp_cnt = [0 for _ in range(len(self.emo_model.buffer))]
                
                self.emo_model.assign_label_update() # 统计一下每个神经元的归属label

                for i in range(len(self.emo_model.assign_label)):
                    temp_cnt[self.emo_model.assign_label[i]] += output[0, i] # 第一个维度是batch, 

                predict_label = torch.argmax(torch.tensor(temp_cnt))
                if self.env.people_is_near():
                    # print(state)
                    if predict_label == 0:
                        print("Posive Emotion!!!")
                    else:
                        print("Negative Emotion!!!")
    def test_for_plot(self):
        print("测试 micc 后的数据准确率")
        if self.all_sit_down_flag:
            self.sit_down_all_the_time()
            self.all_sit_down_flag = False

        test_thread = threading.Thread(target=self._test_for_plot_thread, name="test_for_plot_thread")
        test_thread.start()
        pass
    def _test_for_plot_thread(self):
        while True:
            state = self.env.get_three_human_input()
            
            with torch.no_grad():
                # print(state)
                self.emo_model.state_from_dict(filename=model_path, device=torch.device("cuda"))
                output = self.emo_model.step(state, reward=0)
                
                temp_cnt = [0 for _ in range(len(self.emo_model.buffer))]
                
                self.emo_model.assign_label_update() # 统计一下每个神经元的归属label

                for i in range(len(self.emo_model.assign_label)):
                    temp_cnt[self.emo_model.assign_label[i]] += output[0, i] # 第一个维度是batch, 

                predict_label = torch.argmax(torch.tensor(temp_cnt))
                if self.env.people_is_near():
                    self.test_i +=1 
                    # print(state)
                    if self.test_i > self.test_num:
                        np.savetxt('./cobot_test_acc_before_micc.txt', fmt='%f', delimiter=',',X=self.test_acc_list) # 保存预测成功率曲线
                        break # 线程结束
                    which_human = self.env.which_human_near()
                    print("预测结果为： ", predict_label, " human is :", which_human)
                    if (predict_label == 1 and which_human == 2) or (predict_label == 0 and which_human==3):
                        self.right_i +=1
                    self.test_acc_list.append(self.right_i / (1+self.test_i))
                        



    def do_emotional_action(self):
        # 根据不同的情感，做出不同的动作
        pass

    def sit_down_all_the_time(self):
        temp_thread = threading.Thread(target=self._sit_down_thread,name="_sit_down_thread")
        temp_thread.start()


    def _sit_down_thread(self):
        while(True):
            sit_down(2.0)

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

    