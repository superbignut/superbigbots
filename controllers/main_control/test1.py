# 测试一下使用 spike_first 的简单实现效果
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
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


log_path = './log/episode_reward'
writer = SummaryWriter(log_path)

for file in os.listdir(log_path):
    os.remove(log_path + '/' + file) # 把和上一次的log清空

# tensorboard.exe --logdir=./log/episode_reward

env = gym.make('CartPole-v1', render_mode='human')

state_dim = env.observation_space.shape[0]

action_dim = env.action_space.n

hidden_dim = 20

device = torch.device("cuda")

num_episodes = 5000

run_time = 6
backend = spaic.Torch_Backend(device)
backend.dt = 0.1

time_step = int(run_time / backend.dt)

lr = 0.001

class rnet(spaic.Network):
    def __init__(self):
        super().__init__()

        self.input = spaic.Encoder(num=state_dim, time=run_time, coding_method='poisson', unit_conversion=100) 

        self.layer1 = spaic.NeuronGroup(action_dim, model='lif')

        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full',weight=(np.random.rand(action_dim, state_dim) * 0.1)) 
    
        self.output = spaic.Decoder(num=action_dim, dec_target=self.layer1, time=run_time, coding_method='spike_counts') # spike_counts
        
        self._learner = Learner(algorithm='rstdpet', trainable=self.connection1, run_time=run_time, lr=lr) 

        self.reward = spaic.Reward(num=hidden_dim, dec_target=self.layer1, coding_time=run_time, coding_method='environment_reward', dec_sample_step=time_step, pop_size=1) # 这两个参数不太明白

        """
            dec_sample_step: 是奖励信号的触发时间, 每隔多长时间奖励一次
            
            pop_size: popilation 把pop_size大小的神经元分为一组进行整体操作, 如果使用environment_reward，应该和pop_size没关系
        
        """
        self.mon_V = spaic.StateMonitor(self.layer1, 'V', nbatch=False)
        self.mon_O = spaic.SpikeMonitor(self.layer1, 'O', nbatch=False)
        
        self._backend = backend


env.reset(seed=0)

agent = rnet()

agent_2 = rnet()




class r_adaptive:
    def __init__(self) -> None:
        
        pass

    def get_reward(self, state):
        pass


r_agent = r_adaptive()
for i in range(10):
    for i_episode in tqdm(range(num_episodes // 10)):

        state,_ = env.reset()
        done = False
        episode_reward = 0
        reward = 0
        while not done:
            
            agent.input(torch.tensor(np.array([state]), device=device))  # 第一个维度是batch
            
            agent.reward(0)# 传递奖励信号

            agent.run(run_time) # rstdp学习

            output = agent.output.predict # 


            
            print(output[0]) # [0] 是batch维度

            next_state, reward, done, _, _ = env.step(output[0].argmax().item())

            state = next_state
            episode_reward += reward
        

        plt.subplot(2,2,1)
        time_line = agent.mon_V.times  # 取layer1层时间窗的坐标序号
        value_line0 = agent.mon_V.values[0][0] # batch、神经元编号
        plt.plot(time_line, value_line0, label='V')

        plt.scatter(agent.mon_O.spk_times[0], agent.mon_O.spk_index[0], s=40, c='g', label='Spike') # agent.mon_O.spk_times[0] 这个维度应该是batch
        plt.ylabel("action_0 Membrane potential")
        plt.ylim((-0.1, 1.5))
        plt.xlabel("time")



        plt.subplot(2,2,2)
        value_line1 = agent.mon_V.values[0][1] 
        plt.plot(time_line, value_line1, label='V')
        plt.scatter(agent.mon_O.spk_times[0], agent.mon_O.spk_index[0], s=40, c='r', label='Spike')
        plt.ylabel("action_1 Membrane potential")
        plt.ylim((-0.1, 1.5))
        plt.xlabel("time")
        plt.show()

        # plt.scatter(output_line_time, output_line_index, s=40, c='r', label='Spike')

        # writer.add_scalar('episode_reward', episode_reward, i_episode + i * num_episodes // 10)











