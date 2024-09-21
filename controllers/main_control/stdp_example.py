import os
from SPAIC import spaic
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from SPAIC.spaic.Learning.Learner import Learner
from SPAIC.spaic.IO.Dataset import CUSTOM_MNIST
from SPAIC.spaic.IO.Dataset import MNIST as dataset
from SPAIC.spaic.Library.Network_saver import network_save

# 参数设置

# 设备设置
SEED = 0
np.random.seed(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
print(device)
backend = spaic.Torch_Backend(device)
backend.dt = 0.1
sim_name = backend.backend_name
sim_name = sim_name.lower()
# 创建训练数据集

root = './SPAIC/spaic/Datasets/MNIST'
train_set = CUSTOM_MNIST(root, is_train=True) # 这里加一个mast 只要01
test_set =CUSTOM_MNIST(root, is_train=False)

run_time = 256 * backend.dt # 这么长的时间吗
node_num = 784
label_num = 100
bat_size = 1 # 每次只拿一个数据出来的吗？


train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)


class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        self.input = spaic.Encoder(num=node_num, time=run_time, coding_method='poisson', unit_conversion=0.6375) # 就是给发放速率乘了因子,from 论文

        self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex') # stdp_ex 比 stdp_ih 多了一层阈值的衰减 \tao_\theta, 论文中有提到这个自适应的阈值
        self.layer2 = spaic.NeuronGroup(label_num, model='lifstdp_ih') # 维度都是100， 

        self.output = spaic.Decoder(num=label_num, dec_target=self.layer1, time=run_time, coding_method='spike_counts') # layer1作为了输出层, 兴奋次数作为输出

        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=(np.random.rand(label_num, 784) * 0.3)) # 100 * 784
        
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full',# 这里应该是one-to-one, 因为是对角连接，所以就相当于是 1-1 连接了吗
                                              weight=    np.diag(np.ones(label_num)) * 22.5   ) 
        
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', 
                                              weight=(   np.ones((label_num, label_num)) - np.diag(np.ones(label_num))    ) * (-120)) # 起到一个抑制的作用，除了1-1的前一个，侧向抑制，并起到竞争的效果

        # self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1, run_time=run_time) # 这里也只是训练 从输入到第一层的连接，其余层不变
        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1, run_time=run_time) # 这里也只是训练 从输入到第一层的连接，其余层不变

        self.reward = spaic.Reward(num=label_num, dec_target=self.layer1, coding_time=run_time, coding_method='environment_reward', dec_sample_step=1) 
        #
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        
        self.set_backend(backend)


Net = TestNet()
Net.build(backend)

print("Start running")

eval_losses = []
eval_acces = []
losses = []
acces = []
spike_output = [[]] * 10

im = None

for epoch in range(1):
    # 训练阶段
    pbar = tqdm(total=len(train_loader))
    train_loss = 0
    train_acc = 0

    for i, item in enumerate(train_loader): # batch_size 是 2
        
        data, label = item
        print(data.shape)
        Net.input(data)
        # Net.output(label)
        Net.reward(1)
        Net.run(run_time) # 

        output = Net.output.predict # 输出维度 1 * 100

        if spike_output[label[0]] == []:
            spike_output[label[0]] = [output]
        else:
            spike_output[label[0]].append(output) # 每次把输出放在相应的label下面

        if sim_name == 'pytorch':
            label = torch.tensor(label, device=device, dtype=torch.long)

        im = Net.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white',
                     reshape=True, n_sqrt=int(np.sqrt(label_num)), side=28, im=im, wmax=1) #把权重 画出来 100 * 784 = 100 * 28 * 28

        pbar.update()

    a = [sum(spike_output[i]) / len(spike_output[i]) for i in range(len(spike_output))] #  spike_output 应该只有10个 0-9 这些数字, 然后把 所有 1*100 的权重相加后取平均
    # 所以某个label， 比如 = 0， 对应的在某些位置，如果总是1的话，sum之后会积累的很大

    assign_label = torch.argmax(torch.cat((a), 0), 0) # cat 之后 从 10 个 1*100 变成 10 *100
    # argmax 找到 100个位置里 这个位置对应的，在哪一个lable 中占比最高，生成一个 100 的tensor

    # 测试阶段
    eval_loss = 0
    eval_acc = 0
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict

            if sim_name == 'pytorch':
                label = torch.tensor(label, device=device, dtype=torch.long)
            spike_output_test = [[]] * 10 # 再按照 train的方法统计一遍
            for o in range(assign_label.shape[0]): # 100
                if spike_output_test[assign_label[o]] == []:
                    spike_output_test[assign_label[o]] = [output[:, o]] # 每次把o位置的输出放在 这个位置label 对应的[]中，最后看哪个label对应的多，就是预测的结果是哪个
                else:                       
                    spike_output_test[assign_label[o]].append(output[:, o]) # 这里如果再引入奖励的话，再根据结果的是否正确，利用这次观测的数据，对网络进行一次调节，也可以对assign_label 的统计buffer 进行一次更新

            test_output = []
            for o in range(len(spike_output_test)):                             # 实际中可能再去考虑一下，emotion 变化的稳定性的问题 基本就完美了
                if spike_output_test[o] == []:
                    pass
                else:
                    test_output.append([sum(spike_output_test[o]) / len(spike_output_test[o])])

            predict_label = torch.argmax(torch.tensor(test_output, device=label.device))
            num_correct = (predict_label == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc

            pbarTest.update()


    pbarTest.close()
    print('epoch:{},Test Acc:{:.4f}'
          .format(epoch, eval_acc / len(test_loader)))
    print("")
