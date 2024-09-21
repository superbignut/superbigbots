"""
    使用spaic的自带的rstdp学习模块进行测试，模块中有 RSTDP、 RSTDPET、RewardSwitchSTDP 三种奖励算法

    前两个是出自于 Reinforcement learning through modulation of spike-timing-dependent synaptic plasticity - 2007

    ipynb中实现的Rstdp出自First-Spike-Based Visual Categorization Using Reward-Modulated STDP - 2017 使用两套网络的思想也很有帮助

    这里用来做个什么任务，来测试一下算法的效果呢，强化学习的环境吗，倒立摆？
"""
import os
from SPAIC import spaic
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from SPAIC.spaic.Learning.Learner import Learner
from SPAIC.spaic.IO.Dataset import MNIST as dataset
from SPAIC.spaic.Library.Network_saver import network_save



if __name__ =='__main__':

    print("yes")