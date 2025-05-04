import pygame
import sys
import os
import json
import tkinter as tk
import torch
from chess_board import ChessBoardWindow

# 初始化 tkinter，用于后续配置保存（这里仅用于初始化，不再调用文件对话框）
tk_root = tk.Tk()
tk_root.withdraw()

# 初始化pygame
pygame.init()

# 设置窗口大小和标题
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("中国象棋")

# 定义常用颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)

# 使用支持中文的字体（WenQuanYi Zen Hei），字号可根据需要调整
FONT = pygame.font.SysFont("WenQuanYi Zen Hei", 36)

# 定义按钮类，方便以后扩展和管理
class Button:
    def __init__(self, text, rect, callback, bg_color=GRAY, hover_color=DARK_GRAY, text_color=BLACK):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.callback = callback
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = FONT

    def draw(self, surface):
        # 根据鼠标位置改变按钮颜色
        mouse_pos = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.bg_color
        pygame.draw.rect(surface, color, self.rect)
        # 渲染按钮文字
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        # 处理鼠标点击事件
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.callback()


# 为未来功能预留接口——这些函数可在后续开发中完善

def launch_pvc_game():
    # 启动人机对战界面：调用棋盘窗口类的 run 方法
    board_window = ChessBoardWindow(headless=False,mcts_simulations=50)
    board_window.run()

def launch_pvp_game():
    print("启动玩家对战界面（未实现）")
    # 这里预留玩家对战界面的启动接口

def launch_settings():
    # 改进后的设置界面：主动扫描模型文件夹并使用下拉框选择
    settings_width, settings_height = 800, 600
    settings_screen = pygame.display.set_mode((settings_width, settings_height))
    pygame.display.set_caption("设置界面")
    
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    FONT_SMALL = pygame.font.SysFont("WenQuanYi Zen Hei", 24)
    
    # 指定模型文件夹
    model_folder = "models"
    files = [f for f in os.listdir(model_folder) if f.endswith(".pt")]
    if not files:
        files = ["无模型文件"]
    selected_index = 0

    # 默认MCTS模拟次数
    mcts_simulations = 100  
    user_input = str(mcts_simulations)
    input_active = False
    drop_down_active = False
    
    # 定义控件区域
    model_rect = pygame.Rect(50, 50, 300, 50)
    mcts_rect = pygame.Rect(50, 120, 300, 50)
    back_rect = pygame.Rect(50, 190, 300, 50)
    
    clock = pygame.time.Clock()
    running = True
    while running:
        settings_screen.fill(WHITE)

        
        # 绘制模型下拉框主区域
        pygame.draw.rect(settings_screen, BLACK, model_rect, 2)
        current_file = files[selected_index]
        current_text = FONT_SMALL.render("模型: " + current_file, True, BLACK)
        settings_screen.blit(current_text, (model_rect.x + 10, model_rect.y + 10))
        # 绘制下拉箭头
        arrow_points = [(model_rect.right - 20, model_rect.y + 20),
                        (model_rect.right - 10, model_rect.y + 20),
                        (model_rect.right - 15, model_rect.y + 30)]
        pygame.draw.polygon(settings_screen, BLACK, arrow_points)
        
               # 绘制 MCTS 模拟次数输入区域
        pygame.draw.rect(settings_screen, BLACK, mcts_rect, 2)
        mcts_text = FONT_SMALL.render("MCTS模拟次数: " + user_input, True, BLACK)
        settings_screen.blit(mcts_text, (mcts_rect.x + 10, mcts_rect.y + 10))
        
        # 绘制 返回按钮
        pygame.draw.rect(settings_screen, BLACK, back_rect, 2)
        back_text = FONT_SMALL.render("返回主菜单", True, BLACK)
        settings_screen.blit(back_text, (back_rect.x + 10, back_rect.y + 10))
       # 如果下拉框激活，则显示所有选项
        if drop_down_active:
            option_height = 30
            # 新增：先绘制一个不透明背景区域
            drop_rect = pygame.Rect(model_rect.x, model_rect.bottom, model_rect.width, option_height * len(files))
            pygame.draw.rect(settings_screen, WHITE, drop_rect)
            for i, file in enumerate(files):
                option_rect = pygame.Rect(model_rect.x, model_rect.bottom + i * option_height, model_rect.width, option_height)
                pygame.draw.rect(settings_screen, WHITE, option_rect)  # 确保每个选项背景为白色
                pygame.draw.rect(settings_screen, BLACK, option_rect, 1)
                option_text = FONT_SMALL.render(file, True, BLACK)
                settings_screen.blit(option_text, (option_rect.x + 10, option_rect.y + 5))
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if model_rect.collidepoint(event.pos):
                    # 切换下拉状态
                    drop_down_active = not drop_down_active
                elif drop_down_active:
                    option_height = 30
                    for i in range(len(files)):
                        option_rect = pygame.Rect(model_rect.x, model_rect.bottom + i * option_height, model_rect.width, option_height)
                        if option_rect.collidepoint(event.pos):
                            selected_index = i
                            drop_down_active = False
                            break
                elif mcts_rect.collidepoint(event.pos):
                    input_active = True
                elif back_rect.collidepoint(event.pos):
                    running = False
                else:
                    drop_down_active = False
            elif event.type == pygame.KEYDOWN:
                if input_active:
                    if event.key == pygame.K_RETURN:
                        try:
                            mcts_simulations = int(user_input)
                        except ValueError:
                            print("输入无效")
                        input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        user_input = user_input[:-1]
                    else:
                        user_input += event.unicode
        
        pygame.display.flip()
        clock.tick(30)
    
    # 当设置界面退出时，将最终设置保存到配置文件
    agent_model_path = os.path.join(model_folder, files[selected_index])
    print("最终设置：")
    print("训练模型路径：", agent_model_path)
    print("MCTS模拟次数：", mcts_simulations)
    
    config = {
        "agent_model_path": agent_model_path,
        "mcts_simulations": mcts_simulations
    }
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("配置已保存到 config.json")
    
    # 恢复主菜单显示模式
    pygame.display.set_mode((WIDTH, HEIGHT))

def exit_game():
    pygame.quit()
    sys.exit()

# 根据窗口大小和按钮数量，计算按钮的排版
button_width = 200
button_height = 50
button_spacing = 20
total_height = 4 * button_height + 3 * button_spacing
start_y = (HEIGHT - total_height) // 2

# 创建各个按钮对象
buttons = [
    Button("人机对战", ((WIDTH - button_width) // 2, start_y, button_width, button_height), launch_pvc_game),
    Button("玩家对战", ((WIDTH - button_width) // 2, start_y + button_height + button_spacing, button_width, button_height), launch_pvp_game),
    Button("设置", ((WIDTH - button_width) // 2, start_y + 2 * (button_height + button_spacing), button_width, button_height), launch_settings),
    Button("退出", ((WIDTH - button_width) // 2, start_y + 3 * (button_height + button_spacing), button_width, button_height), exit_game)
]

# 主菜单循环
def main_menu():
    clock = pygame.time.Clock()
    while True:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game()
            for button in buttons:
                button.handle_event(event)
        
        # 填充背景色
        SCREEN.fill(WHITE)
        
        # 绘制标题
        title_font = pygame.font.SysFont("WenQuanYi Zen Hei", 48)
        title_surf = title_font.render("中国象棋", True, BLACK)
        title_rect = title_surf.get_rect(center=(WIDTH // 2, start_y - 60))
        SCREEN.blit(title_surf, title_rect)
        
        # 绘制所有按钮
        for button in buttons:
            button.draw(SCREEN)
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main_menu()
