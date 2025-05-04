import argparse
import time
from mcts import Mcts
from PolicyValueNet import PolicyValueNet
from chessboard import Chess_board,action_labels,action_label2i
from self_play import selfplay  # 假定 selfplay 类已定义
from mcts import board_to_ezDecode
from PolicyValueNet import state_to_positions
from collections import deque
import numpy as np
from ai_vs_people import AiHumanPlay

import torch
class CChessTrainer:
    def __init__(self, train_epoch, batch_size,mcts_playout_times):
        self.buffer_size = 10000
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.data_buffer = deque(maxlen=self.buffer_size) 
        self.log_file = open("training_log.txt", "w")
        self.board = Chess_board()
        self.PV_net = PolicyValueNet(num_gpus=1, num_of_res_block=9)
        self.kl_target = 0.025
        self.learning_rate = 0.001 
        self.lr_multiplier = 1
        self.epoch = 5
        self.playout_times = mcts_playout_times
        
        def mirror_action_label(label):
            fx, fy, tx, ty = label[0], label[1], label[2], label[3]
            def mirror_x(ch):
                return chr(ord('i') - (ord(ch) - ord('a')))
            return mirror_x(fx) + fy + mirror_x(tx) + ty
        
        # 只做一次：生成 mirror_index 列表
        self.mirror_index = []
        for i, lbl in enumerate(action_labels):
            mirrored_lbl = mirror_action_label(lbl)
            self.mirror_index.append(action_label2i.get(mirrored_lbl, -1))
        # mirror_index[i] = j 意味着 action i 的镜像动作是 j；-1 表示无镜像

    def mirror_probs(self, probs):
        """用预先计算好的 self.mirror_index 快速做镜像"""
        mirrored = np.zeros_like(probs)
        for i, j in enumerate(self.mirror_index):
            if j >= 0:
                mirrored[j] = probs[i]
        return mirrored
    def selfplay(self):
        # 返回 selfplay 数据和每局步数
        sp = selfplay(play_times=1, mcts_search_round=self.playout_times, temperature=1.0, num_gpu=1,PV_net=self.PV_net)
        return sp.selfplay_n_times()

    def policy_update(self):
        import random
        import torch.nn.functional as F

        # 数据不足则不更新
        if len(self.data_buffer) < self.batch_size:
            return

        # 随机采样一个 mini-batch
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # 分离数据
        state_batch = np.array([data[0] for data in mini_batch])       # shape: [batch, 9, 10, 14]
        mcts_probs_batch = np.array([data[1] for data in mini_batch])    # shape: [batch, 2086]
        winner_batch = np.array([data[2] for data in mini_batch])        # shape: [batch,]
        winner_batch = np.expand_dims(winner_batch, 1)                   # shape: [batch, 1]

        start_time = time.time()


        # 计算旧的策略概率，先将模型置为 eval 模式,算kl散度
        self.PV_net.eval()
        with torch.no_grad():
            old_policy_logits, old_v = self.PV_net.forward(state_batch)
            # 转为 softmax 概率
            old_probs = F.softmax(old_policy_logits, dim=1).detach().cpu().numpy()
        self.PV_net.train()

        # 对同一 mini-batch 进行多轮更新
        for epoch in range(self.epoch):
            # 执行一次训练更新
            acc, loss, global_step = self.PV_net.train_step((state_batch, mcts_probs_batch, winner_batch),self.learning_rate*self.lr_multiplier)
            
            # 更新后计算新的策略概率
            self.PV_net.eval()
            with torch.no_grad():
                new_policy_logits, new_v = self.PV_net.forward(state_batch)
                new_probs = F.softmax(new_policy_logits, dim=1).detach().cpu().numpy()
            self.PV_net.train()

            # 避免概率中出现0
            old_probs_safe = np.where(old_probs > 1e-10, old_probs, 1e-10)
            new_probs_safe = np.where(new_probs > 1e-10, new_probs, 1e-10)
            # 计算 KL 散度
            kl = np.mean(np.sum(old_probs_safe * (np.log(old_probs_safe) - np.log(new_probs_safe)), axis=1))
            
            # 若 KL 散度过大则提前停止本 mini-batch 的训练更新
            if kl > self.kl_target * 4:
                break

        print("[Policy-Value-Net] -> Training Took {} s".format(time.time() - start_time))
        print("[Policy-Value-Net] -> Global Step: {}, Accuracy: {:.4f}, Loss: {:.4f}".format(global_step, acc, loss))

        if kl > self.kl_target * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_target / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(winner_batch - old_v.squeeze().numpy()) / np.var(winner_batch)
        explained_var_new = 1 - np.var(winner_batch - new_v.squeeze().numpy()) / np.var(winner_batch)

        print(
            "[Policy-Value-Net] -> KL Divergence:{}; \t\tLr Multiplier:{}; \n[Policy-Value-Net] -> Loss:{}; \t\t\tAccuracy:{}; \n[Policy-Value-Net] -> Explained Var (old):{}; \t\t\tExplained Var (new):{};\n".format(
                kl, self.lr_multiplier, loss, acc, explained_var_old, explained_var_new))
        

    def get_mirror_data(self,state, mcts_prob, winner):
        mcts_prob = mcts_prob.copy()
        # 生成镜像数据
        mirror_state = np.flip(state, axis=0)
        mirror_mcts_prob = self.mirror_probs(mcts_prob)
        mirror_winner = -winner
        return mirror_state, mirror_mcts_prob, mirror_winner
        



    def run(self):
        batch_iter = 0
        start_time = time.time()
        print("[Train CChess] -> Training Start ({} Epochs)".format(self.train_epoch))
        try:
            total_data_len = 0
            while batch_iter <= self.train_epoch:
                batch_iter += 1
                game_begin_time = time.time()
                print(f"game {batch_iter} begin")
                play_data, episode_len = self.selfplay()
                print(f"game {batch_iter}/ end,which takes {time.time() - game_begin_time} seconds")
                print("[Train CChess] -> Batch {}/{}; Episode Length: {}; Iteration: {}".format(
                    batch_iter, self.train_epoch, episode_len, batch_iter))
                extend_data = []
                for state, mcts_prob, winner in play_data[0]:
                    states_data = state_to_positions(state)
                    extend_data.append((states_data, mcts_prob, winner))
                    extend_data.append(self.get_mirror_data(states_data, mcts_prob, winner))
                self.data_buffer.extend(extend_data)
                total_data_len += len(extend_data)
                self.log_file.write("time: {} \t total_data_len: {}\n".format(time.time() - start_time, total_data_len))
                self.log_file.flush()
                print("training data_buffer len:", len(self.data_buffer))
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
            self.log_file.close()
            self.PV_net.save()
            print("[Train CChess] -> Training Finished, Took {} s".format(time.time() - start_time))
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving model...")
            self.log_file.close()
            self.PV_net.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'selfplay'], default='train')
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_playout',type=int,default=400)
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = CChessTrainer(train_epoch=args.train_epoch, batch_size=args.batch_size,mcts_playout_times=args.train_playout)
        trainer.run()

