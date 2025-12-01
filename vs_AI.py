import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import marubatu5 as m
import game 
import time
                

num_states = 25         # 盤のマス数
num_actions = 25       # 行動数

# NN
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # Q(s, ·)

# バッファ
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = map(list, zip(*batch))
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buf)
    
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

env = m.TicTacToe5x5()

# パラメータ
global_step = 0
target_update_interval = 1000 
gamma = 0.95


# ---------------------------------------------------------------------------------
# 繰り返し
while True:
    s_t = env.reset()
    done = False
    done_AI = False
    player_skip = False

    while not done:
        # playerの処理
        # Check if game ended
        if done_AI:
            print("\n" + "=" * 50)
            if env.winner == 0:
                print("Draw!")
            elif env.winner == 1:
                print("You Win!")
            else:
                print("AI Wins!")
            print("=" * 50)
            
            restart = input("\nPlay again? (y/n): ").strip().lower()                
            if restart == 'y':
                done = True
                break
            else:
                exit()

        if not player_skip:
            # Human player's turn
            action = game.get_player_action()
            time.sleep(0.5)
            
            # Handle special commands
            if action == 'quit':
                print("Exiting game...")
                break
            
            if action == 'restart':
                print("Restarting game...")
                state = env.reset()
                continue
            
            if action is None:
                # Invalid input, retry
                continue
            
            # Execute action
            s_t, reward, done, info = env.step(action)
            
            if not info['valid_move']:
                print("Invalid move! Please choose an empty square.")
                continue
            
            # Check if game ended
            if done:
                # env.display()
                print("\n" + "=" * 50)
                if env.winner == 0:
                    print("Draw!")
                elif env.winner == 1:
                    print("You Win!")
                else:
                    print("AI Wins!")
                print("=" * 50)
                
                restart = input("\nPlay again? (y/n): ").strip().lower()                
                if restart == 'y':
                    done = True
                    break
                else:
                    exit()

        # Qネットの処理
        a_t = select_action(s_t)
        time.sleep(0.5)  

        # バッファの処理 & 環境を進める
        s_next, reward, done_AI, info = env.step(a_t)
        print(f'reward = {reward}')
        if not info["valid_move"]:
            player_skip = True
        else:
            player_skip = False
# ---------------------------------------------------------------------------------
