import numpy as np
import torch
idx2role = {
    1: '兵',
    2: '炮',
    3: '车',
    4: '马',
    5: '相',
    6: '士',
    7: '帅',
    -1: '卒',
    -2: '炮',
    -3: '车',
    -4: '马',
    -5: '象',
    -6: '士',
    -7: '将'
}

def create_action_labels():
    """
    函数里想要生成一个“广义”的可能走法列表（不考虑阻挡，也不细分红黑阵营区域）
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    Advisor_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                      'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    Bishop_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                     'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                     'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                     'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']

    for l1 in range(9):
        for n1 in range(10):
            destinations = [(t, n1) for t in range(9)] + \
                           [(l1, t) for t in range(10)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(9) and n2 in range(10):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
#士走斜线
    for p in Advisor_labels:
        labels_array.append(p)
#大象走田
    for p in Bishop_labels:
        labels_array.append(p)

    return labels_array



# 根据走法生成镜像动作
def flip_action_label(param):
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in param]

# this array is for storing the action labels
labels_array = create_action_labels()
labels_len = len(labels_array)

# this array is for storing the flipped action labels,that is action_labels when the side is flipped
flipped_labels = flip_action_label(labels_array)

#index function is for getting the index of the flipped action labels,
# this means you can input flipped action index and get the standard action index

unflipped_index = [labels_array.index(x) for x in flipped_labels]


def flip_policy(prob):
    prob = np.squeeze(prob) # .flatten()
    return np.asarray([prob[ind] for ind in unflipped_index])

action_labels=create_action_labels()
action_i2label = {i: val for i, val in enumerate(action_labels)}
action_label2i = {val: i for i, val in enumerate(action_labels)}

class Chess_board(object):
    def __init__(self):
        self.Piece2Pos = {
            3: [(0, 0), (0, 8)],  # 车
            4: [(0, 1), (0, 7)],  # 马
            5: [(0, 2), (0, 6)],  # 相
            6: [(0, 3), (0, 5)],  # 士
            7: [(0, 4)],  # 帅
            2: [(2, 1), (2, 7)],  # 炮
            1: [(3, 0), (3, 2), (3, 4), (3, 6), (3, 8)],  # 兵

            -3: [(9, 0), (9, 8)],  # 车
            -4: [(9, 1), (9, 7)],  # 马
            -5: [(9, 2), (9, 6)],  # 象
            -6: [(9, 3), (9, 5)],  # 士
            -7: [(9, 4)],  # 将
            -2: [(7, 1), (7, 7)],  # 炮
            -1: [(6, 0), (6, 2), (6, 4), (6, 6), (6, 8)]  # 卒
        }
        self.Pos2Piece = {pos: piece for piece, positions in self.Piece2Pos.items() for pos in positions}
        self.turn = 'r'
        self.over = False
        self.consecutive_non_kill = 0
        self.winner = None

        #定死的

    def reset_board(self):
        self.Piece2Pos = {
            3: [(0, 0), (0, 8)],  # 车
            4: [(0, 1), (0, 7)],  # 马
            5: [(0, 2), (0, 6)],  # 相
            6: [(0, 3), (0, 5)],  # 士
            7: [(0, 4)],  # 帅
            2: [(2, 1), (2, 7)],  # 炮
            1: [(3, 0), (3, 2), (3, 4), (3, 6), (3, 8)],  # 兵

            -3: [(9, 0), (9, 8)],  # 车
            -4: [(9, 1), (9, 7)],  # 马
            -5: [(9, 2), (9, 6)],  # 象
            -6: [(9, 3), (9, 5)],  # 士
            -7: [(9, 4)],  # 将
            -2: [(7, 1), (7, 7)],  # 炮
            -1: [(6, 0), (6, 2), (6, 4), (6, 6), (6, 8)]  # 卒
        }
        self.Pos2Piece = {pos: piece for piece, positions in self.Piece2Pos.items() for pos in positions}
        self.turn = 'r'
        self.over = False
        self.consecutive_non_kill = 0
        self.winner = None
    #R（车）、N（马）、B（象 / 相）、A（士 / 仕）、
    # K（将 / 帅）、C（炮）、P（卒 / 兵）。
    def encode_chessboard(self):
            """
            将当前的象棋棋盘转换为 FEN (ICCS) 字符串表示。
            返回值示例（初始布局）:
            RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr
            """
            
            # 定义一个辅助函数，将整数棋子编码为 FEN 字符
            def piece_to_fen(piece):
                # 红方（正数） => 大写
                # 黑方（负数） => 小写
                mapping = {
                    1: 'P',   2: 'C',   3: 'R',
                    4: 'N',   5: 'B',   6: 'A',  7: 'K'
                }
                abs_val = abs(piece)
                base_char = mapping[abs_val]
                return base_char if piece > 0 else base_char.lower()
            
            rows = []
            
            # 棋盘是 10 行（0~9），每行 9 列（0~8）
            for row in range(10):
                empty_count = 0
                row_str = ""
                for col in range(9):
                    if (row, col) in self.Pos2Piece:
                        # 如果之前累积了空格数，先写出来
                        if empty_count > 0:
                            row_str += str(empty_count)
                            empty_count = 0
                        
                        # 放上棋子符号
                        piece = self.Pos2Piece[(row, col)]
                        row_str += piece_to_fen(piece)
                    else:
                        # 该位置为空，累计
                        empty_count += 1
                
                # 如果行尾仍有空位没记录，则补上
                if empty_count > 0:
                    row_str += str(empty_count)
                
                rows.append(row_str)
            
            # 用斜杠拼接每行
            fen_string = "/".join(rows)
            return fen_string
    
    def decode_chess_chess_board(self, encoded_board):
        """
        将 FEN (ICCS) 字符串解析回当前的棋盘布局。
        """
        # 先定义一个小工具，把 FEN 中的单个字母解析成对应的棋子代号
        def fen_to_piece(char):
            """
            大写 => 红方 (正数)
            小写 => 黑方 (负数)
            """
            fen_map = {
                'P': 1, 'C': 2, 'R': 3,
                'N': 4, 'B': 5, 'A': 6, 'K': 7
            }
            val = fen_map[char.upper()]
            return val if char.isupper() else -val

        ranks = encoded_board.split('/')
        if len(ranks) != 10:
            raise ValueError("FEN 字符串必须包含 10 行，用 '/' 分割。")

        # 清空原数据，用新的布局覆盖
        self.Pos2Piece = {}

        for row in range(10):
            rank = ranks[row]
            col = 0
            for ch in rank:
                if ch.isdigit():
                    # 数字表示连续的空格
                    col += int(ch)
                else:
                    # 字母表示一个棋子
                    piece = fen_to_piece(ch)
                    if col < 9:  # 合法列
                        self.Pos2Piece[(row, col)] = piece
                    col += 1

            # 若某行解析结束后列数不等于 9，说明 FEN 非法或缺失
            if col != 9:
                raise ValueError(f"第 {row+1} 行解析后列数不是 9，FEN 不合法。")

        # 更新 Piece2Pos
        self.Piece2Pos = {}
        for pos, piece in self.Pos2Piece.items():
            self.Piece2Pos.setdefault(piece, []).append(pos)
    def move(self, action):
        """
        根据 action("e2e3" 之类)，将起点处的棋子移动到终点。
        如果终点有敌方棋子则执行吃子动作，并清零 self.consecutive_non_kill；
        反之则自增 (无吃子计数)。
        如果被吃或将要吃的是将/帅(-7或7)，则游戏结束(self.over = True)。
        移动完成后，切换回合。
        """
        # 将 'a0b0' 形式的起终点解析为数字坐标,方式不同
        start_pos = (int(action[1]),ord(action[0]) - ord('a'))
        end_pos   = (int(action[3]),ord(action[2]) - ord('a'))

        # 1) 检查起点是否有棋子
        if start_pos not in self.Pos2Piece:
            raise ValueError(f"起始位置 {start_pos} 没有棋子，无法移动。")

        piece = self.Pos2Piece[start_pos]

        # 2) 如果终点上有棋子，先把它移除(吃子)
        captured_piece = None
        if end_pos in self.Pos2Piece:
            captured_piece = self.Pos2Piece[end_pos]

            # 如果被吃的是将/帅，则游戏结束
            if captured_piece == 7 or captured_piece == -7:
                self.over = True
                self.winner = 'b' if captured_piece > 0 else 'r'
            # 从 Piece2Pos 移除被吃的棋子
            self.Piece2Pos[captured_piece].remove(end_pos)
            if not self.Piece2Pos[captured_piece]:
                del self.Piece2Pos[captured_piece]

        # 3) 移动：用新棋子占领 end_pos 并清空 start_pos
        self.Pos2Piece[end_pos] = piece
        del self.Pos2Piece[start_pos]

        # 4) 更新 Piece2Pos 中该棋子的位置信息
        self.Piece2Pos[piece].remove(start_pos)
        self.Piece2Pos[piece].append(end_pos)

        # 5) 根据是否吃子更新 consecutive_non_kill
        if captured_piece is not None:
            self.consecutive_non_kill = 0
        else:
            self.consecutive_non_kill += 1

        # 6) 切换回合
        self.turn = 'b' if self.turn == 'r' else 'r'
    def all_legal_moves(self):
        turn=self.turn
        self.encode_chessboard




# 2086
#print(len(create_action_labels()))

