from chessboard import Chess_board,action_label2i
import time
from mcts import Mcts
nonKill_seilling = 60
import numpy  as np
from PolicyValueNet import PolicyValueNet


#selfplay 的棋盘始终保持正向，在传入valuepolicy net需要反转，mcts也是

def try_flip(state, flip=False):
    if not flip:
        return state

    rows = state.split('/')

    #大小写互转
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    #对所有大小写互转
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])
    return "/".join([swapall(row) for row in reversed(rows)])

# selfplay 写完了，利用selfplay可以实现一局对局，
class selfplay():
    def __init__(self,play_times,mcts_search_round,temperature,num_gpu,PV_net):
        self.play_times = play_times
        self.mcts_search_round=mcts_search_round
        self.temperature=temperature
        self.board = Chess_board()
        self.num_gpu = num_gpu
        self.PV_net = PV_net
        self.mcts_tree = Mcts("RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr",self.PV_net,search_threads=8)
    def selfplay_n_times(self):
        data_per_round=[]
        episode_per_round = []
        for i in range(self.play_times):
            self.board.reset_board()
            self.mcts_tree.reload()
            states, mcts_probs, current_players = [], [], []
            z = 0
            winner = 0
            start_time = time.time()

            while not self.board.over:
                move_round = 0
                mcts_res = self.mcts_tree.mcts(self.board.turn,self.mcts_search_round,self.temperature,restrict_round=0)
                action_choosen, action_probs ,win_prob = mcts_res

                #mct的可能动作和对应的prob
                action_label,label_prob =action_probs[0][0],action_probs[0][1]
                encoded_board=self.board.encode_chessboard()
                state= try_flip(encoded_board,self.board.turn == 'b')
                states.append(state)
                action_prob =[0]*2086
                
                # 判断一下当前是什么turn
                if self.board.turn == 'r':
                    for i in range(len(action_label)):
                        action_prob[action_label2i[action_label[i]]] = label_prob[i]
                else:
                    for i in range(len(action_label)):
                        act = "".join((str(9 - int(a)) if a.isdigit() else a) for a in action_label[i])
                        action_prob[action_label2i[act]] = label_prob[i]
                mcts_probs.append(action_prob)
                current_players.append(self.board.turn)

                #last_state = self.game_borad.state
                # move 中包含了切换turn,也包含了判结束，将军和平局都算了
                self.board.move(action_choosen)
                move_round+=1

                if self.board.consecutive_non_kill >= nonKill_seilling:
                    #tie
                    self.board.over = 1
                    self.board.winner = 't'
                    print(f"Self-Play: Round{move_round} Over,taking {time.time()-start_time}")
                #这块代码写的很迷惑
                if self.board.over:
                    z = np.zeros(len(current_players))
                    winner =self.board.winner
                    #获胜方赋值1，失败方赋值-1
                    z[np.array(current_players) == winner] =1.0
                    z[np.array(current_players) != winner]= -1.0

            data_per_round.append(zip(states,mcts_probs,z))
            episode_per_round.append(len(z))
        return data_per_round,episode_per_round
    
            


"""
针对selfplay，输出的states,mcts_probs,z没有任何的方向倾向，可以认为是队长态完全等价的描述
"""