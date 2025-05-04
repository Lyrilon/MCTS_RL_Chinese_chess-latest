from threading import Lock
from collections import deque, defaultdict, namedtuple
from asyncio import Future
import asyncio
from asyncio.queues import Queue
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from PolicyValueNet import input_preprocess
import numpy as np
import torch
from chessboard import flip_policy
from chessboard import action_label2i
from tqdm import trange
from sys import getsizeof

c_PUCT = 5
Nx=9
Ny=10

def validate_move(c, upper=True):
    if (c.isalpha()):
        if (upper == True):
            if (c.islower()):
                return True
            else:
                return False
        else:
            if (c.isupper()):
                return True
            else:
                return False
    else:
        return True


def get_piece_num(state):
    res = 0
    for pos in state:
        if pos.isalpha():
            res +=1
    return res

def is_kill_move(state_prev, state_next):
    return get_piece_num(state_prev) - get_piece_num(state_next)


def check_bounds(toY, toX):
    if toY < 0 or toX < 0:
        return False

    if toY >= Ny or toX >= Nx:
        return False
    return True

def board_to_ezDecode(board):
    board = board.replace("2", "11")
    board = board.replace("3", "111")
    board = board.replace("4", "1111")
    board = board.replace("5", "11111")
    board = board.replace("6", "111111")
    board = board.replace("7", "1111111")
    board = board.replace("8", "11111111")
    board = board.replace("9", "111111111")
    return board.split("/")

def sim_do_action(in_action, in_state):
        x_trans = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}
        src = in_action[0:2]
        dst = in_action[2:4]
        src_x = int(x_trans[src[0]])
        src_y = int(src[1])
        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])
        board_positions = board_to_ezDecode(in_state)
        line_lst = []
        for line in board_positions:
            line_lst.append(list(line))
        lines = np.array(line_lst)
        lines[dst_y][dst_x] = lines[src_y][src_x]
        lines[src_y][src_x] = '1'
        board_positions[dst_y] = ''.join(lines[dst_y])
        board_positions[src_y] = ''.join(lines[src_y])
        board = "/".join(board_positions)
        board = board.replace("111111111", "9")
        board = board.replace("11111111", "8")
        board = board.replace("1111111", "7")
        board = board.replace("111111", "6")
        board = board.replace("11111", "5")
        board = board.replace("1111", "4")
        board = board.replace("111", "3")
        board = board.replace("11", "2")
        return board

def softmax(x):
    # print(x)
    probs = np.exp(x - np.max(x))
    # print(np.sum(probs))
    probs /= np.sum(probs)
    return probs

def create_pos_labels():
    rs = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in range(9):
        for j in range(10):
            move = letters[8-i] + numbers[j]
            rs.append(move)
    return rs


board_pos_name = np.array(create_pos_labels()).reshape(9,10).transpose()

def get_legal_moves(state, current_player):

        moves = []
        k_x = None
        k_y = None
        K_x = None
        K_y = None
        face_to_face = False
        board_positions = np.array(board_to_ezDecode(state))

        
        for y in range(board_positions.shape[0]):
            for x in range(len(board_positions[y])):

                if(board_positions[y][x].isalpha()):

                    if(board_positions[y][x] == 'r' and current_player == 'b'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, Nx):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, Ny):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif(board_positions[y][x] == 'R' and current_player == 'r'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, Nx):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, Ny):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif ((board_positions[y][x] == 'n' or board_positions[y][x] == 'h') and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=False) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=False) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'N' or board_positions[y][x] == 'H') and current_player == 'r'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=True) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=True) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'b' or board_positions[y][x] == 'e') and current_player == 'b'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'B' or board_positions[y][x] == 'E') and current_player == 'r'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'a' and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'A' and current_player == 'r'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'k'):
                        k_x = x
                        k_y = y

                        if(current_player == 'b'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                                upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                        moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'K'):
                        K_x = x
                        K_y = y

                        if(current_player == 'r'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if check_bounds(toY, toX) and validate_move(board_positions[toY][toX],
                                                                                upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                        moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'c' and current_player == 'b'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, Nx):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, Ny):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'C' and current_player == 'r'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, Nx):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, Ny):
                            m = board_pos_name[y][x] + board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'p' and current_player == 'b'):
                        toY = y - 1
                        toX = x

                        if (check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=False)):
                            moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

                        if y < 5:
                            toY = y
                            toX = x + 1
                            if (check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

                            toX = x - 1
                            if (check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

                    elif (board_positions[y][x] == 'P' and current_player == 'r'):
                        toY = y + 1
                        toX = x

                        if (check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=True)):
                            moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

                        if y > 4:
                            toY = y
                            toX = x + 1
                            if (check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

                            toX = x - 1
                            if (check_bounds(toY, toX) and validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(board_pos_name[y][x] + board_pos_name[toY][toX])

        if(K_x != None and k_x != None and K_x == k_x):
            face_to_face = True
            for i in range(K_y + 1, k_y, 1):
                if(board_positions[i][K_x].isalpha()):
                    face_to_face = False

        if(face_to_face == True):
            if(current_player == 'b'):
                moves.append(board_pos_name[k_y][k_x] + board_pos_name[K_y][K_x])
            else:
                moves.append(board_pos_name[K_y][K_x] + board_pos_name[k_y][k_x])

        return moves

QueueItem = namedtuple("QueueItem", "feature future")

def game_over(state):
    if state.find('K') == -1:
        return 1
    elif state.find('k') == -1:
        return 2
    return 0
"""
N（访问次数）：node.N 表示该节点被访问的次数。在 MCTS 中，每次选择一个节点进行扩展和模拟时，该节点的访问次数就会增加。这个统计量在计算节点的平均价值（如 Q 值）以及使用 UCB 公式进行节点选择时起着重要作用。例如，UCB 公式中的一部分会考虑节点的访问次数来平衡探索和利用。
W（累计价值）：node.W 表示该节点累计获得的价值。在模拟过程中，当从该节点开始进行一次模拟并得到一个结果（例如游戏的胜负结果）时，这个结果对应的价值会累加到 W 上。节点的平均价值 Q 通常通过 Q = W / N 来计算。
"""
class Mcts():
    def __init__(self, in_state, policy_value_net, search_threads):
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3    #0.03
        self.p_ = (1 - self.noise_eps) * 1 + self.noise_eps * np.random.dirichlet([self.dirichlet_alpha])
        self.root = leaf_node(None, self.p_, in_state)
        self.c_puct = 5    #1.5
        # self.policy_network = in_policy_network
        self.policy_value_net = policy_value_net
        self.node_lock = defaultdict(Lock)

        self.virtual_loss = 3
        self.now_expanding = set()
        self.expanded = set()
        self.cut_off_depth = 30
        # self.QueueItem = namedtuple("QueueItem", "feature future")
        self.thread_remain = asyncio.Semaphore(search_threads)
        self.queue = Queue(search_threads)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.a= 2
        self.exploration_mode = 1
    async def  push_queue(self,features):
        future = self.loop.create_future()
        item = QueueItem(features,future)
        await self.queue.put(item)
        return future

    async def start_tree_search(self,node, current_player, restrict_round):
        """
        在蒙特卡罗树搜索（MCTS）这种多线程或多协程并发执行的场景中，可能会有多个
        协程同时尝试对同一个节点进行操作，比如同时对一个节点进行扩展。为了避免多
        个协程同时对同一个节点进行扩展而导致的数据不一致或错误，代码使用了
        now_expanding 集合来记录当前正在被扩展的节点。
        """
        now_expanding = self.now_expanding
        #当前要拓展的节点正在被别的线程操作，就不断让出权限直到节点空闲

        while node in now_expanding:
            await asyncio.sleep(1e-4)
        
        if node not in self.expanded:
            self.now_expanding.add(node)

            net_input = input_preprocess(node.state,current_player)
            future = await self.push_queue(net_input)

            await future
            # 作为value直接返回
            action_probs,value = future.result()
            action_probs=action_probs.detach().numpy()
            # action_probs, value = self.forward(positions)
            if current_player == 'b':
                action_probs = flip_policy(action_probs)
            
            moves = get_legal_moves(node.state,current_player)

            node.expand(moves,action_probs)
            self.now_expanding.remove(node)
            


            return value[0]*-1
        else:
            """
            expanded finished,we select first
            """
            #select child with maximum action score
            last_state = node.state

            action,node = node.select(c_PUCT)
            current_player = 'r' if current_player == 'b' else 'b'
            cur_state = node.state

            #mcts 对不杀移动的限制

            if is_kill_move(last_state,cur_state) == 0:
                restrict_round += 1 
            else:
                restrict_round = 0
            #切换
            last_state = cur_state

            #virtual loss
            node.N += self.virtual_loss
            node.W += -self.virtual_loss

            end_sign = game_over(last_state)
            if end_sign == 1:
                value = 1.0 if current_player =='b' else -1.0
            elif end_sign == 2:
                value = 1.0 if current_player == 'r' else -1.0
            elif restrict_round >=60:
                value =0.0
            else:
                value = await self.start_tree_search(node,current_player,restrict_round)

            node.N -= self.virtual_loss
            node.W += self.virtual_loss

            node.back_up_v(value)

            return value * -1
    async def tree_search(self,node,cur_player,restrict_round):
        self.running_simulation_num += 1
        #有空闲的sem才开始
        async with self.thread_remain:
            res = await self.start_tree_search(node,cur_player, restrict_round)
            self.running_simulation_num-=1
            return res
        
    async def critic(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            features = np.asarray([np.squeeze(item.feature) for item in item_list])    # asarray
            action_probs, value = self.policy_value_net(features)
            for p, v, item in zip(action_probs, value, item_list):
                item.future.set_result((p, v))

    def Q(self, move) -> float:
        ret = 0.0
        find = False
        for a, n in self.root.child.items():
            if move == a:
                ret = n.Q
                find = True
        if(find == False):
            print("{} not exist in the child".format(move))
        return ret
    def mcts(self,cur_player,search_round,temperature,restrict_round)-> tuple[list,list,int]:
        node  = self.root
        if node not in self.expanded:
            net_input=input_preprocess(node.state,cur_player)
            action_probs , state_value =  self.policy_value_net(net_input)
            action_probs = action_probs.detach().numpy()
            #这里给action_probs做了翻转，没理解为啥            
            if cur_player == 'b':
                action_probs = flip_policy(action_probs)
            #flip的作用是我最后得到的结果是：在反过的动作编码下顺序的prob
            move_set = get_legal_moves(node.state,cur_player)
            node.expand(move_set,action_probs)
            self.expanded.add(node)

        coroutine_set = []
        for _ in range(search_round):
            task = self.tree_search(node,cur_player,restrict_round)
            coroutine_set.append(task)
        coroutine_set.append(self.critic())


        self.loop.run_until_complete(asyncio.gather(*coroutine_set))
        # 获取键列表
        actions = list(self.root.child.keys())
        # 获取值对象的 N 属性列表
        visits = np.array([node.N for node in self.root.child.values()])
        # condition,yes_v,no_v
        visits = np.where(visits > 1.0e-10, visits, 1.0e-10)
        probs = softmax(1.0 / temperature * np.log(visits, where=visits > 0))
        move_prob_list = []
        move_prob_list.append((actions,probs))
        if self.exploration_mode:
            act_choosen =  np.random.choice(actions, p=0.75 * probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
        else:
            act_choosen = np.random.choice(actions, p=probs)
        win_rate = self.Q(act_choosen)
        self.update_tree(act_choosen)
        return act_choosen,move_prob_list,win_rate

    def reload(self):
        self.root = leaf_node(None,self.p_,"RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")
        self.expanded = set()
    def update_tree(self, act):
        self.expanded.discard(self.root)
        self.root = self.root.child[act]
        self.root.parent = None
class leaf_node(object):
    def __init__(self, in_parent, in_prior_p, in_state):
        self.P = in_prior_p
        self.Q = 0
        self.N = 0
        self.v = 0
        self.U = 0
        self.W = 0
        self.parent = in_parent
        self.child = {}
        self.state = in_state
        

    def is_leaf(self):
        return self.child == {}
    
    
    
    def expand(self,moves,action_probs):
        p_sum = 1e-8
        action_probs = np.squeeze(action_probs)

        for action in moves:
            state_after_done = sim_do_action(action,self.state)
            #mov_prob是积累的
            mov_prob = action_probs[action_label2i[action]]
            new_node = leaf_node(self,mov_prob,state_after_done)
            self.child[action] = new_node
            p_sum += mov_prob
        #归一化处理，和为1
        for action,node in self.child.items():
            node.P /=p_sum
    
    
    def back_up_v(self,value):
        self.N += 1
        self.W += value
        self.v = value
        self.Q = self.W / self.N  # node.Q += 1.0*(value - node.Q) / node.N
        self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
    def get_Q_plus_U(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        U = c_puct * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
        return self.Q + U
    def select(self, c_puct):
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U(c_puct))
    
