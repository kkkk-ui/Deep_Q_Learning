import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import syogi as m
from dqn_game import QNetwork
                
    
# 選択
def select_action(s_t):
    s_tensor = torch.tensor(s_t, dtype=torch.float32).view(1, -1)
    q_values = q_net(s_tensor).squeeze(0)  
    a_t = torch.argmax(q_values).item()
    return a_t

# 初期化
q_net = QNetwork()  # モデル構造を作る
q_net.load_state_dict(torch.load("qnet_final.pth", map_location="cpu"))
q_net.eval()  # 推論モード（必須）

MODEL_PATH = 'qnet_final.pth'  # dqn_game.pyで保存されたモデルファイル

env = m.TicTacToe5x5()
waiting_for_player = False
a_t = None

# ---------------------------------------------------------------------------------

def on_click(event):
        if not waiting_for_player:
            return
        
            
        # クリック位置を座標に変換
        x = int(event.xdata)
        y = int(event.ydata)

        a_t = (env.board[y][x]-1) * 25 + (y * 5 + x)
        print(a_t)
        # s_t, reward, done , info = env.step(a_t)


def wait_for_player_move():
    waiting_for_player = True
    print("あなたの番です。盤面をクリックして手を選んでください。")

    while waiting_for_player:
        env.plt.pause(0.1)  # イベントループを回す
    waiting_for_player = False

    return a_t


def play_game():
    env.fig.canvas.mpl_connect('button_press_event', on_click)
    env.reset()

    while env.done is False:
        # プレイヤー1の手番
        wait_for_player_move()
        if env.done:
            break

        # プレイヤー2の手番
        wait_for_player_move()


# ---------------------------------------------------------------------------------
def main():
    """メイン関数"""
    print("\n" + "=" * 50)
    print("DQN 5x5 将棋風ゲーム")
    print("=" * 50 + "\n")
    
    # モード選択
    print("ゲームモードを選択してください:")
    print("1: 交代プレイモード (プレイヤー1 vs プレイヤー2)")
    print("2: AI対戦モード (プレイヤー1 vs AI)")
    
    while True:
        mode_input = input("モードを選択 (1 or 2): ").strip()
        if mode_input == '1':
            print("\n交代プレイモードを選択しました\n")

            play_game()
            break
        elif mode_input == '2':
            print("\nAI対戦モードを選択しました\n")
            
            print("Loading trained DQN agent...")
            q_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

            break
        else:
            print("無効な入力です。1 または 2 を入力してください。")
    
    # エージェントの作成


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()