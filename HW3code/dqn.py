import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game.flappy_bird import GameState
from general import export_plot


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()       # 继承父类nn.Module的init方法

        ##############################################################
        ################### YOUR CODE HERE  Part 1 ###################
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 150000
        self.replay_memory_size = 10000
        self.minibatch_size = 32
        ######################## END YOUR CODE #######################
        ##############################################################

        ##############################################################
        ################### YOUR CODE HERE  Part 2 ###################
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(3136, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)
        ######################## END YOUR CODE #######################
        ##############################################################

    def forward(self, x):
        ##############################################################
        ################### YOUR CODE HERE  Part 3 ###################
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = out.view(out.size(0),-1)
        out = self.fc4(out)
        out = F.relu(self.fc4(out))
        out = self.fc5(out)


        ######################## END YOUR CODE #######################
        ##############################################################
        return out


def init_weights(m):                                        # 初始化权重
    if type(m) == nn.Conv2d or type(m) == nn.Linear:        # 判断m的类型，如果是卷积层或全连接层，就初始化权重
        torch.nn.init.uniform(m.weight, -0.01, 0.01)        # 初始化权重w在-0.01到0.01之间
        m.bias.data.fill_(0.01)                             # 初始化偏置b都为0.01


def image_to_tensor(image):                                 # 将图片转换为torch tensor, 供神经网路读取
    image_tensor = image.transpose(2, 0, 1)                 # 将图片的通道维度放到第一维
    image_tensor = image_tensor.astype(np.float32)          # 将图片数据类型转换为np.float32
    image_tensor = torch.from_numpy(image_tensor)           # 将np.array转换为torch tensor
    if torch.cuda.is_available():                           # 如果CUDA可用，将tensor放到GPU上。对于苹果电脑，使用MPS
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):                             
    image = image[0:288, 0:404]                             # 截取图片中间部分
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)  # 将图片resize为84*84，转换为灰度图
    image_data[image_data > 0] = 255                        # 将灰度图中大于0的像素值设为255，变成黑白图
    image_data = np.reshape(image_data, (84, 84, 1))        # 将图片reshape为84*84*1
    return image_data


def train(model, start, iteration = 0):
    
    ##############################################################
    ################### YOUR CODE HERE  Part 4 ###################
    # 定义Adam优化器，学习率为1e-6
    optimizer = Adam(model.parameters(), lr=1e-6)

    # 使用MSE loss作为损失函数
    criterion = nn.MSELoss()
    ######################## END YOUR CODE #######################
    ##############################################################

    # 初始化游戏状态
    game_state = GameState()

    # 初始化replay memory
    replay_memory = []

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)    # 初始化action都为0
    action[0] = 1                                                           # 第一个action为1
    image_data, reward, terminal = game_state.frame_step(action)            # 获取第一个状态
    image_data = resize_and_bgr2gray(image_data)                            # 将图片resize为84*84，转换为灰度图    
    image_data = image_to_tensor(image_data)                                # 将图片转换为torch tensor
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0) # 将图片复制4份，作为初始状态

    # 初始化epsilon
    epsilon = model.initial_epsilon
    # 初始化总奖励
    total_reward_in_one_game = 0
    all_total_reward = []
    # epsilon的衰减值
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # 开始训练
    while iteration < model.number_of_iterations:

        ##############################################################
        ################### YOUR CODE HERE  Part 5 ###################
        # 神经网络对于当前状态的输出
        output = model(state)[0]

        # 初始化action都为0
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # 如果CUDA可用，将action放到GPU上
            action = action.cuda()

        # epsilon greedy策略
        random_action = random.random() <= epsilon
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # 如果CUDA可用，将action_index放到GPU上
            action_index = action_index.cuda()

        action[action_index] = 1                                          # 将action_index对应的action设为1，作为下一步的action

        image_data_1, reward, terminal = game_state.frame_step(action)    # 采取action后，获取下一个状态
        image_data_1 = resize_and_bgr2gray(image_data_1)                  # 将图片resize为84*84，转换为灰度图
        image_data_1 = image_to_tensor(image_data_1)                      # 将图片转换为torch tensor
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0) # 将state中的后三张图片和新的图片拼接在一起，作为下一个状态

        action = action.unsqueeze(0)                                                 # 将action的维度扩展为
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0) # 将reward转换为torch tensor

        # 将(state, action, reward, state_1, terminal)加入replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # 如果replay memory的长度超过replay_memory_size，就删除最早的数据
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # 减小epsilon
        epsilon = epsilon_decrements[iteration]
        ######################## END YOUR CODE #######################
        ##############################################################

        ##############################################################
        ################### YOUR CODE HERE  Part 6 ###################
        # 从replay memory中随机采样minibatch_size个数据
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # 将minibatch中的数据分别存储到state_batch, action_batch, reward_batch, state_1_batch中
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # 如果CUDA可用，将数据放到GPU上
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # 下一状态的神经网络输出
        output_1_batch = model(state_1_batch)

        # 接下来，计算每个采样状态的q_target，它是一个向量。
        # 其每个元素，如果状态是终止状态，那么是r_j，否则是 r_j + gamma * max_a' Q(s_j+1, a')
        # 这里可以写一个for循环，对于每个状态，计算y_j并加入q_target中；也可以直接用向量运算，计算所有状态的q_target
        q_target = reward_batch + gamma * torch.max(output_1_batch, dim=1)[0] * (1 - torch.tensor([d[4] for d in minibatch], dtype=torch.float32) )
        

        # 计算每个状态的Q(s, a)。q_value是一个向量，每个元素是batch中每个状态的Q值，由神经网络输出得到
        q_value = model(state_batch).gather(1, action_batch)
        ######################## END YOUR CODE #######################
        ##############################################################

        ##############################################################
        ################### YOUR CODE HERE  Part 7 ###################

        # 将q_target从模型中detach, 不需要计算梯度
        q_target = q_target.detach()
        

        # 清空梯度
        optimizer.zero_grad()  

        # 计算loss
        loss = nn.MSELoss()(q_value, q_target)

        # 反向传播计算梯度
        loss.backward() 

        # 执行一步参数更新
        optimizer.step()


        ######################## END YOUR CODE #######################
        ##############################################################

        state = state_1                                     # 将state更新为state_1
        iteration += 1                                      # 更新iteration
        total_reward_in_one_game += reward.numpy()[0][0]    # 更新总奖励

        if iteration % 25000 == 0:                         # 每25000次迭代保存一次模型
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        if terminal:                                       # 如果游戏结束，打印当前迭代次数，总奖励，epsilon
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "total reward:", total_reward_in_one_game)
            all_total_reward.append(total_reward_in_one_game)
            total_reward_in_one_game = 0

        if iteration % 2000 == 0:                          # 每2000次迭代画一次结果图
            export_plot(all_total_reward, "Total Rewards", "results/result_"+ str(iteration) + ".png")


def test(model):                    # 测试模型
    game_state = GameState()        # 初始化游戏状态

    # 初始化第一次，这里与train中一样
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]    # 获取神经网络的输出

        # 这里得到当前状态下的action，与train中一样，action是一个one-hot向量
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # 如果CUDA可用，将action放到GPU上
            action = action.cuda()
        
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # 如果CUDA可用，将action_index放到GPU上
            action_index = action_index.cuda()
        action[action_index] = 1

        # 采取action后，获取下一个状态，也与train中一样
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # 更新state
        state = state_1


def main(mode):            # 主函数
    cuda_is_available = torch.cuda.is_available()     # 判断CUDA是否可用

    if mode == 'test':                                # 如果mode是test，加载模型并测试
        model = torch.load(                           # 加载模型，如果CUDA可用，将模型放到GPU上
            'pretrained_model/current_model_1000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  
            model = model.cuda()

        test(model)                                   # 测试模型

    elif mode == 'train':                            # 如果mode是train，训练模型
        if not os.path.exists('pretrained_model/'):  # 如果pretrained_model文件夹不存在，创建文件夹
            os.mkdir('pretrained_model/') 

        model = NeuralNetwork()                      # 初始化模型

        if cuda_is_available:                        # 如果CUDA可用，将模型放到GPU上
            model = model.cuda()

        model.apply(init_weights)                    # 初始化模型的权重
        start = time.time()                          # 记录开始时间

        train(model, start)                          # 训练模型

    elif mode == 'cont_train':                       # 如果mode是cont_train，加载模型并继续训练
        model = torch.load(                          # 加载模型，如果CUDA可用，将模型放到GPU上
            'pretrained_model/current_model_750000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        model.number_of_iterations = 1000000         # 更新迭代次数

        if cuda_is_available:                        # 如果CUDA可用，将模型放到GPU上
            model = model.cuda()

        start = time.time()                          # 记录开始时间

        train(model, start, 750000)


if __name__ == "__main__":                           # 运行主函数,需要传入参数，train, test, cont_train其中之一
    main(sys.argv[1])
