import numpy as np

'''
状态字符编码方式介绍：
首先使用KARBNPCkarbnpc表示棋子
In English ,we see KARBNPCkarbnpc as king, advisor, rook, bishop, knight, pawn, cannon
炮和兵的编码字母是 C和 P
并且在遇到连续空位置的时候用数字表示，数字表示连续的空位置的个数
在换行的时候使用"/"表示


字符串编码->考虑黑方->展开数字占空位-> 14通道特征图->输入
'''


# from string with "2" to "11", "3" to "111", ..., "9" to "111111111",which means unfold a compressed board
def replace_board_tags(board):
    board = board.replace("2", "11")
    board = board.replace("3", "111")
    board = board.replace("4", "1111")
    board = board.replace("5", "11111")
    board = board.replace("6", "111111")
    board = board.replace("7", "1111111")
    board = board.replace("8", "11111111")
    board = board.replace("9", "111111111")
    return board.replace("/", "")

pieces_order = 'KARBNPCkarbnpc' # 9 x 10 x 14

# ind is for mapping the pieces to the index.means 14 kinds of pieces,which ultimately transfered to 14 channels
# from 'KARBNPCkarbnpc' to 0-13
ind = {pieces_order[i]: i for i in range(14)}


# we firstly transform the board state to a string
def state_to_positions(state):
    board_state = replace_board_tags(state)
    pieces_plane = np.zeros(shape=(9, 10, 14), dtype=np.float32)
    for row in range(10):
        for col in range(9):
            v = board_state[row * 9 + col]
            if v.isalpha():
                pieces_plane[col][row][ind[v]] = 1

    assert pieces_plane.shape == (9, 10, 14)
    return pieces_plane

# 返回十四张特征图



def input_preprocess(state,player):
    def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a
        #对所有大小写互转
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])
    # 黑方对于状态的处理，需要翻转大小写，与此同时
    if player == 'b':
        rows = state.split('/')
        state = "/".join([swapall(row) for row in reversed(rows)])
    state = state_to_positions(state)
    state = np.expand_dims(state,0)
    return state

# 测试input_preprocess函数
if __name__ == "__main__":
    # 测试用例
    state = "RNBA1ABNR/9/1C5C1/1P1P1P1P1/9/9/9/9/9/1p1p1p1p1/1c5c1/9/rnbakabnr"
    player = 'r'
    result = input_preprocess(state, player)
    print(result.shape)  # 输出 (1, 9, 10, 14)
    # 输出第一张特征图
    print(result[0, :, :, 0])  # 输出第一张特征图


import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def raise_error(file, line, message ,color="red"):
    if color == "red":
        pass
        # print('\033[31m<USER ERROR> <File: %s> <Line: %s>\n\t%s<Msg: %s>\033[0m'%(file, line, " "*5, message))
    else:
        print('\033[1;33;1m<USER ERROR> <File: %s> <Line: %s>\n\t%s<Msg: %s>\033[0m'%(file, line, " "*5, message))


# 定义残差块模块,此处的momentum是BN的momentum，用于BN的滑动平均

'''
momentum的设置问题：
1. momentum=0.9: 适用于大批量训练，滑动平均较快，适合训练阶段。
2. momentum=0.99: 适用于小批量训练，滑动平均较慢，适合推理阶段。
3. momentum=0.999: 适用于小批量训练，滑动平均更慢，适合推理阶段。
理由：
使用较大的momentum值（如0.9）时，BN层的均值和方差会更快地适应当前批次的数据分布，这在训练阶段是有利的，因为训练数据通常是动态变化的。
使用较小的momentum值（如0.99或0.999）时，BN层的均值和方差会更慢地适应当前批次的数据分布，这在推理阶段是有利的，因为推理数据通常是静态的。
举个例子，假设我们在训练一个图像分类模型，使用了BN层来加速训练。在训练阶段，我们可能会使用较大的批量大小（如256），此时可以使用momentum=0.9来快速适应当前批次的数据分布。
而在推理阶段，我们可能会使用较小的批量大小（如1），此时可以使用momentum=0.99或0.999来更慢地适应当前批次的数据分布，以提高模型的稳定性和准确性。
'''
class ResidualBlock(nn.Module):
    def __init__(self, filters, momentum):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters, eps=1e-5, momentum=momentum)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters, eps=1e-5, momentum=momentum)
    
    def forward(self, x):
        orig = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + orig
        out = F.relu(out)
        return out

class PolicyValueNet(nn.Module):
    def __init__(self, num_gpus, num_of_res_block=9):
        super(PolicyValueNet, self).__init__()
        print("init gpu net")
        self.save_path = "./gpu_models"
        self.logging = True
        self.num_gpus = num_gpus

        if not os.path.exists(self.save_path):
            raise_error(__file__, sys._getframe().f_lineno,
                        message=("save path -> %s does not exists" % (self.save_path)))
            os.makedirs(self.save_path)
        
        # 网络结构参数
        self.filters_size = 128       # 对应 tf 中的 self.filters_size
        self.prob_size = 2086
        self.l2 = 0.0001
        self.momentum = 0.99
        self.global_step = 0

        # 注意：TensorFlow 使用 channels_last ([9, 10, 14])，而 PyTorch 使用 channels_first
        # 第一层卷积：输入通道 14，输出 channels=self.filters_size，kernel_size=3，padding=1（保持尺寸）
        self.conv1 = nn.Conv2d(14, self.filters_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.filters_size, eps=1e-5, momentum=self.momentum)

        # 构建残差块
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(self.filters_size, self.momentum) for _ in range(num_of_res_block)]
        )

        # 策略头：先经过一个 1x1 卷积，输出通道数 2，然后 BatchNorm + ReLU，
        # 接着将 [batch, 2, 9, 10] 展平为 [batch, 9*10*2]，并全连接到输出维度 self.prob_size (2086)
        self.policy_conv = nn.Conv2d(self.filters_size, 2, kernel_size=1, stride=1, padding=0)
        self.policy_bn = nn.BatchNorm2d(2, eps=1e-5, momentum=self.momentum)
        self.policy_fc = nn.Linear(9 * 10 * 2, self.prob_size)

        # 价值头：先经过 1x1 卷积，输出通道数 1，然后 BatchNorm + ReLU，
        # 展平为 [batch, 9*10*1]，全连接到 256 个神经元（ReLU激活），再全连接到 1 个输出，并用 tanh 激活
        self.value_conv = nn.Conv2d(self.filters_size, 1, kernel_size=1, stride=1, padding=0)
        self.value_bn = nn.BatchNorm2d(1, eps=1e-5, momentum=self.momentum)
        self.value_fc1 = nn.Linear(9 * 10 * 1, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # 优化器设置：SGD，学习率 0.001，momentum 0.9，使用 Nesterov
        self.learning_rate = 0.001
        self.momentum_opt = 0.9
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate,
                                   momentum=self.momentum_opt, nesterov=True)

        # 设置保存检查点的路径
        self.checkpoint_dir = os.path.join(self.save_path, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')

        # 尝试加载最新检查点
        start_time = time.time()
        latest_ckpt = self._latest_checkpoint(self.checkpoint_dir)
        if latest_ckpt is not None:
            self.load(latest_ckpt)
            print(latest_ckpt+" loaded")
        print("**************************************************")
        print("Restore Took {} s".format(time.time() - start_time))
        print("**************************************************")

    def _latest_checkpoint(self, checkpoint_dir):
        a = os.getcwd() 
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not files:
            return None
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        return os.path.join(checkpoint_dir, files[-1])

    def save(self):
        checkpoint_path = self.checkpoint_prefix + "_step_{}.pt".format(self.global_step)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)

    def forward(self, positions):
        """
        positions 输入可以是 numpy 数组，形状为 [9,10,14] 或 [batch,9,10,14]，
        我们转换为 [batch,14,9,10] 再进行前向传播。
        """
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        
        if positions.dim() == 3:
            # [9,10,14] -> 转置为 [14,9,10] 并增加 batch 维度
            positions = positions.permute(2, 0, 1).unsqueeze(0)
        elif positions.dim() == 4:
            # [batch,9,10,14] -> 转为 [batch,14,9,10]
            positions = positions.permute(0, 3, 1, 2)

        x = self.conv1(positions)
        x = self.bn1(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.reshape(policy.size(0), -1)  # 展平 [batch, 9*10*2]
        policy_logits = self.policy_fc(policy)     # 输出 shape [batch, prob_size]

        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)        # 展平 [batch, 9*10*1]
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))     # 输出 shape [batch, 1]

        return policy_logits, value

    def compute_metrics(self, pi_, policy_logits):
        # 计算分类准确率：比较预测（argmax(logits)）与目标概率分布的 argmax
        pred = torch.argmax(policy_logits, dim=1)
        target = torch.argmax(pi_, dim=1)
        correct = (pred == target).float().mean()
        return correct.item()

    def apply_regularization(self):
        # L2 正则项：对所有参数求平方和
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2 * l2_loss

    def compute_loss(self, pi_, z_, policy_logits, value):
        with torch.no_grad():
            # 注意：pi_ 为目标策略分布，z_ 为目标价值，均应为 torch 张量
            pass  # 占位，仅在当前作用域内使用名称“loss”
        # 策略损失：使用交叉熵形式（计算 -sum(pi * log_softmax(policy_logits)) 的均值）
        policy_loss = -torch.mean(torch.sum(pi_ * F.log_softmax(policy_logits, dim=1), dim=1))
        # 价值损失：均方误差
        value_loss = F.mse_loss(value, z_)
        # 正则化损失
        l2_loss = self.apply_regularization()
        loss = policy_loss + value_loss + l2_loss
        return loss, policy_loss, value_loss, l2_loss

    def train_step(self, batch, learning_rate=0):
        """
        batch 是 (positions, pi, z)
        positions: 输入棋局状态，形状 [batch,9,10,14] 或 [9,10,14]
        pi: 目标策略分布，形状 [batch, prob_size]
        z: 目标价值，形状 [batch, 1] 或 [batch]
        """
        positions, pi, z = batch
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        if isinstance(pi, np.ndarray):
            pi = torch.from_numpy(pi).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        if z.dim() == 1:
            z = z.unsqueeze(1)

        self.optimizer.zero_grad()
        policy_logits, value = self.forward(positions)
        loss, policy_loss, value_loss, l2_loss = self.compute_loss(pi, z, policy_logits, value)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        acc = self.compute_metrics(pi, policy_logits)
        avg_loss = loss.item()
        return acc, avg_loss, self.global_step

    def inference(self, positions):
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(positions)
        self.train()
        return policy_logits, value


