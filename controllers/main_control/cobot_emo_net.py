"""
    这里主要是类似stdp_minist 来实现一个 EMONET 包括网络的搭建，预训练，buffer的投票和根据交互

    结果进行buffer 的反馈调节等

    这里相较于minist 会多一步 数据的预处理，因为目前如果真要使用过去的10个时间段的数据作为输入的话，其实

    是有一点冗余或者说

    使用积极、消极 代替 happy sad 的说法似乎更书面一点

"""

import collections
import numpy as np
from tqdm import tqdm
import os
import random
import torch
from SPAIC import spaic
import torch.nn.functional as F
from SPAIC.spaic.Learning.Learner import Learner
from SPAIC.spaic.Library.Network_saver import network_save
from SPAIC.spaic.Library.Network_loader import network_load
from SPAIC.spaic.IO.Dataset import MNIST as dataset
from SPAIC.spaic.IO.Dataset import CUSTOM_MNIST, NEW_DATA_MNIST
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

log_path = './log/emo_net'
writer = SummaryWriter(log_path)

for file in os.listdir(log_path):
    os.remove(log_path + '/' + file) # 把和上一次的log清空

device = torch.device("cuda")

node_num = 12
label_num = 100 
bat_size = 1 # 最外层遍历，无实际效果

backend = spaic.Torch_Backend(device)
backend.dt = 0.1 # 步长
run_time = 256 * backend.dt # 运行时间

time_step = int(run_time / backend.dt) # 总共有多少个阻塞突触的时间步，可以用于 rstdp的奖励频率设定

data_time_len = 10
data_kind_len = 6

model_path = 'save/cobot'

buffer_path = 'cobot_buffer.pth'


mic_model_path = 'save/mic_cobot'

mic_buffer_path = 'mic_cobot_buffer.pth'
class EMONET(spaic.Network):
    def __init__(self):
        super().__init__()

        self.input = spaic.Encoder(num=node_num, time=run_time, coding_method='poisson', unit_conversion=0.6) # 这里需要修改

        self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex') # stdp_ex 比 stdp_ih 多了一层阈值的衰减 \tao_\theta, 论文中有提到这个自适应的阈值
        
        self.layer2 = spaic.NeuronGroup(label_num, model='lifstdp_ih') # 维度都是100， 

        self.output  = spaic.Decoder(num=label_num, dec_target=self.layer1, time=run_time, coding_method='spike_counts') # layer1作为了输出层, 兴奋次数作为输出

        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=(np.random.rand(label_num, node_num) * 0.3)) # 100 * 4
        
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=np.diag(np.ones(label_num)) * 22.5   )
        
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', weight=( np.ones((label_num, label_num)) - np.diag(np.ones(label_num)) ) * (-120)) 
        # 起到一个抑制的作用，除了1-1的前一个，侧向抑制，并起到竞争的效果

        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1, run_time=run_time) # 这里也只是训练 从输入到第一层的连接，其余层不变

        self.reward = spaic.Reward(num=label_num, dec_target=self.layer1, coding_time=run_time, coding_method='environment_reward', dec_sample_step=1) # 采样频率是每个时间步一次
        
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        
        self.set_backend(backend)

        self.buffer = [[],[]] # 积极, 消极 投票神经元的buffer # 这里可以改成 【queue， queue】， 但是需要重新训练

        self.assign_label = None

    def step(self, data, reward=1):

        self.input(data) # 输入数据

        self.reward(1) # 这里1，rstdp 退化为stdp 输入奖励 0 则不更新权重

        self.run(run_time) # 前向传播

        return self.output.predict # 输出 结果


    def get_emotion(self, state):
        # 从emo网络 获得emo输出
        return 0
    
    def get_reward(self):
        # 从reward 网络获得 reward输出
        pass
    
    def update(self):
        # 根据交互结果进行更新reward
        pass

    def pre_train(self, data=None, label=None):
        # 这里需要对网路进行预训练
        output = self.step(data)
        self.buffer[label].append(output)
        # print(label, " buffer len is ",len(self.buffer[label]))
        return output
    """     def micc_train(self,data=None, label=None):
        output = self.step(data)
        self.assign_label_update(newoutput=output, newlabel=label) # 把新数据加进去，然后更新 """



    def pre_train_over_save(self):
        self.save_state(filename = model_path) # 这里需要手动删除保存的文件夹
        torch.save(self.buffer, buffer_path) # buffer 也需要保存起来

    def mic_change_over_save(self):
        self.save_state(filename = mic_model_path) # 这里需要手动删除保存的文件夹
        torch.save(self.buffer, mic_buffer_path) # buffer 也需要保存起来

    def assign_label_update(self, newoutput=None, newlabel=None, weight=0):
        # 如果没有新的数据输入，则就是对 assign_label 进行一次计算，否则 会根据权重插入新数据，进而计算
        if newoutput != None:
            self.buffer[newlabel].append(newoutput)
        avg_buffer = [sum(self.buffer[i][-100:]) / len(self.buffer[i][-100:]) for i in range(len(self.buffer))] # sum_buffer 是一个求和之后 取平均的tensor  n * 1 * 100
        # 这里可以只使用 后面的数据进行统计  比如[-300:]
        # avg_buffer = [sum_buffer[i] / len(agent.buffer[i]) for i in range(len(agent.buffer))]
        assign_label = torch.argmax(torch.cat(avg_buffer, 0), 0) # n 个1*100 的list在第0个维度合并 -> n*100的tensor, 进而在第0个维度比哪个更大, 返回一个1维的tensor， 内容是index，0-n， 目前是0和1
        # 这里的 100 个 0 和1 也就代表了， 当前那个神经元 可以代表的 类别是什么
        self.assign_label = assign_label # 初始化结束

if __name__ == '__main__':
    
    net = EMONET()

    for i in tqdm(range(1000)):

        data = np.random.random([1,4])

        print(net.step(data))

