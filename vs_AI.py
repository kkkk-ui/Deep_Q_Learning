import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import syogi as m
from dqn_game import QNetwork
import matplotlib.pyplot as plt
                

MODEL_PATH = 'qnet_final_1000.pth'  # dqn_game.pyで保存されたモデルファイル
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------------
class Game:
    def __init__(self, env, q_net):
        self.waiting_for_player = False
        self.click_koma = False
        self.click_index = False
        self.a_t = None
        self.env = env
        self.q_net = q_net

    def _reverse(self, x, y):
        return 4 - x, 4 - y
        
    def on_click(self, event):
            if not self.waiting_for_player:
                return            
            
            if event.button == 3:
                self.eval()
                return
                
            # クリック位置を座標に変換
            x = int(event.xdata)
            y = int(event.ydata)

            print(f"Clicked at: ({x}, {y})")

            if self.env.current_player == 2:
                x, y = self._reverse(x, y)

            if self.click_koma == False:
                # 駒を選択
                self.selected_koma = self.env.board[y][x]
                self.click_koma = True
                print(f"Selected piece at: {self.selected_koma}")
                return
            
            elif self.click_index == False:
                # 移動先を選択
                self.selected_index = x + y * 5
                self.click_index = True
                print(f"Selected index: {self.selected_index}")

            if self.click_koma and self.click_index:
                # 行動を決定
                self.a_t = (self.selected_koma-1) * 25 + self.selected_index
                self.click_koma = False
                self.click_index = False

                print(f"a_t: {self.a_t}")

                if self.a_t >= 125 or self.a_t < 0:
                    print("無効な手です。もう一度選択してください。")
                    return
                
                self.waiting_for_player = False
                
    def act(self):
        if self.env.current_player == 1:
            print("プレイヤー1の番です。")
        else:
            print("プレイヤー2の番です。")
            
        self.waiting_for_player = True
        while self.waiting_for_player:
            plt.pause(0.1)  # イベントループを回す


    def start(self):
        self.env.reset()
        self.env.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def eval(self):
        s_t = self.env.get_state()
        self.select_action(s_t)

        if self.env.current_player == 2:
            index = self.a_t % 25
            x,y = index % 5, index // 5
            x, y = self._reverse(x, y)
            index = x + y * 5
            koma = (self.a_t // 25) + 1
            

        elif self.env.current_player == 1:
            index = self.a_t % 25
            koma = (self.a_t // 25) + 1
        
        print(f"{self.print_koma(koma)} move to {index}")


    def solo_play(self):
        done = False
        while not done:
            self.act()

            s_t, reward, done, info = self.env.step(self.a_t)  
            if not info['valid_move']:
                print("無効な手です。もう一度選択してください。")
                continue

            if done:
                break

        if info['winner'] == 0:
            print("引き分けです。")
        elif info['winner'] == 1:
            print("プレイヤー1の勝ちです。")
        else:
            print("プレイヤー2の勝ちです。")

    def ai_play(self):
        done = False
        while not done:
            if self.env.current_player == 1:
                self.act()
                s_t, reward, done, info = self.env.step(self.a_t)  
                if not info['valid_move']:
                    print("無効な手です。もう一度選択してください。")
                    continue
            else:
                self.select_action(self.env.get_state())
                s_t, reward, done, info = self.env.step(self.a_t)  
                
            if done:
                break

        if info['winner'] == 0:
            print("引き分けです。")
        elif info['winner'] == 1:
            print("プレイヤーの勝ちです。")
        else:
            print("AIの勝ちです。")

    def select_action(self, s_t):
        #ε=0
        s_tensor = torch.tensor(s_t, dtype=torch.float32).view(1, -1)
        q_values = self.q_net(s_tensor).squeeze(0)  
                
        valid_actions = self.env.get_valid_actions()
        # 2. マスクを作成
        N = q_values.size(-1) # 行動空間のサイズ (例: 25や100など)
        mask = torch.full((N,), -float('inf')).to(device) # まず全てを負の無限大で初期化
        # 有効な行動のQ値は0になるようにマスクを設定
        mask[valid_actions] = 0

        # 3. マスクをQ値に適用
        # q_values_masked = q_values + mask  # Q値とマスクの要素ごとの加算
        # ※ Q値がバッチ形式の場合: q_values + mask.unsqueeze(0)

        # Q値が単一の行動セットであると仮定
        q_values_masked = q_values + mask
        self.a_t = torch.argmax(q_values_masked).item()

    def print_koma(self, koma):
        if koma == 0:
            return "-"
        elif koma == 1:
            return "O"
        elif koma == 2:
            return "X"
        elif koma == 3:
            return "☆"
        elif koma == 4:
            return "∆"
        elif koma == 5:
            return "□" 



# ---------------------------------------------------------------------------------
def main():     
    # モデルの読み込み
    q_net = QNetwork()
    q_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    q_net.eval()


    env = m.TicTacToe5x5()
    game = Game(env, q_net)
    game.start()

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
            game.solo_play()
            break

        elif mode_input == '2':
            print("\nAI対戦モードを選択しました\n")
            game.ai_play()
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