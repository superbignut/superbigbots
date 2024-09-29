"""
    当下的思路是先使用01 来对数据进行stdp预训练，训练的结果要满足对01数据可以达到比较好的分类

    进而切换到更多数据，比如加入"5", 这里假设认为5 归属于1， 那么需要设计网络的权重更新，使得5

    对应的脉冲输出可以不断的替代掉老的数据， 这里总感觉有一个buffer 的含义，

    并且，对真实交互的数据的重要性要进行方法，这里可以体现在，reward上，也可以体现在 buffer的统计上

    感觉本质上还是一个特征提取的操作

    如果只使用原来的算法，很可能最开始的预训练的速度很慢，可以想办法加快一点，对stdp进行修改

    1的脉冲次数 要小于0的次数很多， 这里怀疑可能跟 1 的图像面积小有关系

    这里的面积的大小与出现的次数还没有关系
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


log_path = './log/test2'
writer = SummaryWriter(log_path)

for file in os.listdir(log_path):
    os.remove(log_path + '/' + file) # 把和上一次的log清空

# tensorboard.exe --logdir=./log/episode_reward

root = './SPAIC/spaic/Datasets/MNIST'
train_set = CUSTOM_MNIST(root, is_train=True) # 这里加一个mast 只要01
test_set =CUSTOM_MNIST(root, is_train=False)
new_data_set = NEW_DATA_MNIST(root, is_train=False) # 引入新的数据 包括012
device = torch.device("cuda")


node_num = 784
label_num = 100 # 这里要不了这么多

bat_size = 1


train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)
newdata_loader = spaic.Dataloader(new_data_set,batch_size=bat_size,shuffle=False)

backend = spaic.Torch_Backend(device)
backend.dt = 0.1
run_time = 256 * backend.dt 

time_step = int(run_time / backend.dt)

lr = 0.001 # stdp暂时不用学习率， 真实的学习率应该是体现在算法里面了


class TestNet(spaic.Network):
    def __init__(self):
        super().__init__()

        self.input = spaic.Encoder(num=node_num, time=run_time, coding_method='poisson', unit_conversion=0.6375) # 就是给发放速率乘了因子,from 论文

        self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex') # stdp_ex 比 stdp_ih 多了一层阈值的衰减 \tao_\theta, 论文中有提到这个自适应的阈值
        
        self.layer2 = spaic.NeuronGroup(label_num, model='lifstdp_ih') # 维度都是100， 

        self.output = spaic.Decoder(num=label_num, dec_target=self.layer1, time=run_time, coding_method='spike_counts') # layer1作为了输出层, 兴奋次数作为输出

        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=(np.random.rand(label_num, 784) * 0.3)) # 100 * 784
        
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=np.diag(np.ones(label_num)) * 22.5   )
        
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', weight=( np.ones((label_num, label_num)) - np.diag(np.ones(label_num)) ) * (-120)) # 起到一个抑制的作用，除了1-1的前一个，侧向抑制，并起到竞争的效果

        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1, run_time=run_time) # 这里也只是训练 从输入到第一层的连接，其余层不变

        self.reward = spaic.Reward(num=label_num, dec_target=self.layer1, coding_time=run_time, coding_method='environment_reward', dec_sample_step=1) # 采样频率是每个时间步一次
        #
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        
        self.set_backend(backend)

        self.buffer = [[],[]] # 0, 1 投票神经元的buffer

        self.assign_label = None

    def step(self, data, reward=1):

        self.input(data) # 输入数据

        self.reward(1) # 这里1，rstdp 退化为stdp 输入奖励 0 则不更新权重

        self.run(run_time) # 前向传播

        return self.output.predict # 输出 结果
    

agent = TestNet() # 这里如果一个网络 不行，也可以考虑使用第二个网络进行辅助？

im = None

file_name = 'save/testidea' # 权重的保存文件
buffer_name = 'buffer.pth' # buffer的保存文件

train_num = 1000 # 800次的时候可以达到0.65的成功率 这里应该想办法怎么既能减小 训练次数又能 增大预训练的效率

def TestNet_Init(flag='train'):
    """
        这个函数完成对0 1数据的初始化操作， 也就是希望网路可以达到对0 1的初步分类作用
        可以的话，最好把权重也保存一下，下次再使用的时候就不用再次训练了
    """
    global im
    if flag =='train':

        print("开始预训练")
        for episode in range(1):
            pbar = tqdm(total=len(train_loader))

            for i, item in enumerate(train_loader):
                data, label = item
                output = agent.step(data)
                print(output)
                agent.buffer[label[0]].append(output) # label[0]的第一个维度是batch
                im = agent.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white',
                            reshape=True, n_sqrt=int(np.sqrt(label_num)), side=28, im=im, wmax=1) #把权重 画出来 100 * 784 = 100 * 28 * 28
                pbar.update()
                if i > train_num:
                    break #   
        # network_save(Net=agent,filename='saveidea',save_weight=True)

        agent.save_state(filename=file_name) # 这里需要手动删除保存的文件夹
        torch.save(agent.buffer, buffer_name) # buffer 也需要保存起来

    elif flag == 'load':
        print("加载预训练数据和buffer")
        agent.state_from_dict(filename=file_name, device=device)
        agent.buffer = torch.load(buffer_name)

    assign_label_update() 
    

def assign_label_update(newoutput=None, newlabel=None, weight=0):
    # 如果没有新的数据输入，则就是对 assign_label 进行一次计算，否则 会根据权重插入新数据，进而计算
    if newoutput != None:
        agent.buffer[newlabel].append(newoutput)
    avg_buffer = [sum(agent.buffer[i]) / len(agent.buffer[i]) for i in range(len(agent.buffer))] # sum_buffer 是一个求和之后 取平均的tensor  n * 1 * 100
    # avg_buffer = [sum_buffer[i] / len(agent.buffer[i]) for i in range(len(agent.buffer))]
    assign_label = torch.argmax(torch.cat(avg_buffer, 0), 0) # n 个1*100 的list在第0个维度合并 -> n*100的tensor, 进而在第0个维度比哪个更大, 返回一个1维的tensor， 内容是index，0-n， 目前是0和1
    # 这里的 100 个 0 和1 也就代表了， 当前那个神经元 可以代表的 类别是什么
    agent.assign_label = assign_label # 初始化结束



eval_acc = []

def TestNet_Eval(eval_num = 50):
    # 这里测试一下对0和1的准确率
    if eval_num ==0:
        print("跳过检测环节")
        return
    global im
    right_predict = 0
    
    print("开始检测训练效果")

    with torch.no_grad(): # 这里存粹的测试，不会更新权重 没有 drop层 所以与使用 eval 是一样的效果
        pbar = tqdm(total=len(test_loader))
        for index, item in enumerate(test_loader):
            data, label = item
            output = agent.step(data)
        
            temp_cnt = [0 for _ in range(len(agent.buffer))]
            # 这里的每一个输出[1,2,3,4,5...], 就像是在投票，由于每一个位置类别已经确定，现在，只需要把每个类别的投票的总次数加起来，即可
            for i in range(len(agent.assign_label)):
                temp_cnt[agent.assign_label[i]] += output[0, i] # 第一个维度是batch, 

            predict_label = torch.argmax(torch.tensor(temp_cnt))
            # print("... ", label, predict_label)
            if label[0] == predict_label:
                right_predict += 1
            eval_acc.append(right_predict / (1+index))
            if index > eval_num:
                break
            pbar.update()
    print(" 总数为：", eval_num+2 ," 预测成功为：",right_predict)
    np.savetxt('./eval_acc.txt', fmt='%f', delimiter=',',X=eval_acc)


nd_eval_acc = []

def ND_TestNet_Eval_And_Train(nd_eval_num = 80):
    # 从这里开始增加2 作为新数据，并把2 归属与 1 的label
    global im
    right_predict = 0
    
    print("新数据集测试与微调")

    pbar = tqdm(total=len(newdata_loader))
    for index, item in enumerate(newdata_loader):
        data, label = item
        if label[0] != 0:
            label[0] = 1 # 把2 也归为1
        output = agent.step(data)
        im = agent.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white',
                            reshape=True, n_sqrt=int(np.sqrt(label_num)), side=28, im=im, wmax=1) #把权重 画出来 100 * 784 = 100 * 28 * 28

        temp_cnt = [0 for _ in range(len(agent.buffer))] # 维度是2
        # 这里的每一个输出[1,2,3,4,5...], 就像是在投票，由于每一个位置类别已经确定，现在，只需要把每个类别的投票的总次数加起来，即可
        for i in range(len(agent.assign_label)):
            temp_cnt[agent.assign_label[i]] += output[0, i] # 第一个维度是batch, 

        predict_label = torch.argmax(torch.tensor(temp_cnt))
        # print("... ", label, predict_label)
        if label[0] == predict_label:
            right_predict += 1

        # 这里要使用真实标签对数据对投票集进行补充或修改
        assign_label_update(output, label[0], weight=0) # 权重的体现，可以是乘在output上，也可以是 copy n个插入buffer

        # 这里准备Todo 如果预测错误，可以采取一些迅速的补救措施

        # else

        # 预测正确和错误也可以 不断调整 更新的权重

        nd_eval_acc.append(right_predict / (1+index))
        if index > nd_eval_num:
            break
        pbar.update()

    print(" 总数为：", nd_eval_num+2 ," 预测成功为：", right_predict)
    np.savetxt('./nd_eval_acc.txt', fmt='%f', delimiter=',',X=nd_eval_acc)

def load_run():
    TestNet_Init('load') # 训练
    TestNet_Eval(eval_num=100) # 预测
    ND_TestNet_Eval_And_Train(nd_eval_num=2000) # 加入新数据微调

def train_run():
    TestNet_Init('train') # 训练
    TestNet_Eval() # 预测
    ND_TestNet_Eval_And_Train() # 加入新数据微调
    
def just_train():
    TestNet_Init('train') # 训练

if __name__ == '__main__':
    load_run()
    




""" 
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

 """


