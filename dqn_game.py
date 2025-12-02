import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import syogi as m
import time
                

num_states = 25         # 盤のマス数
num_actions = 125       # 行動数

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
def select_action(s_t, epsilon):
    s_tensor = torch.tensor(s_t.copy(), dtype=torch.float32).view(1, -1)
    q_values = q_net(s_tensor).squeeze(0)  
    if np.random.rand() < epsilon:
        # 探索
        a_t = np.random.randint(num_actions)
    else:
        # 活用
        a_t = torch.argmax(q_values).item()
    return a_t

# 初期化
q_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
buffer = ReplayBuffer()

env = m.TicTacToe5x5()

# パラメータ
global_step = 0
target_update_interval = 1000 
batch_size = 32
gamma = 0.95
num_episodes = 10000

epsilon_start = 1.0
epsilon_end   = 0.05   
epsilon_decay = 0.995  

epsilon = epsilon_start


# ---------------------------------------------------------------------------------
# 繰り返し
for epi in range(num_episodes):
    s_t = env.reset()
    done = False
    player_skip = False

    while not done:
        global_step += 1
        print("iter=", global_step)
        # playerの処理

        if not player_skip:
            # Human player's turn
            action = env.step_random()
            time.sleep(0.005)
            
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
                    print(f'reward = {reward}')
                    s_next_flat = np.array(s_t.copy(), dtype=np.float32).reshape(-1)
                    buffer.push(s_flat, a_t, reward, s_next_flat, done)
                else:
                    print("AI Wins!")
                print("=" * 50)
                
                # restart = input("\nPlay again? (y/n): ").strip().lower()
                restart = "y"
                if restart == 'y':
                    done = True
                    break
                else:
                    exit()

        # Qネットの処理
        a_t = select_action(s_t, epsilon)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        time.sleep(0.005)  

        # バッファの処理 & 環境を進める
        s_next, reward, done, info = env.step(a_t)
        print(f'reward = {reward}')
        if not info["valid_move"]:
            player_skip = True
        else:
            player_skip = False

        s_flat = np.array(s_t.copy(), dtype=np.float32).reshape(-1)
        s_next_flat = np.array(s_next.copy(), dtype=np.float32).reshape(-1)
        buffer.push(s_flat, a_t, reward, s_next_flat, done)

        # Tネットの処理
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states      = torch.tensor(states.copy(), dtype=torch.float32)       
            actions     = torch.tensor(actions, dtype=torch.int64)        
            rewards     = torch.tensor(rewards, dtype=torch.float32)      
            next_states = torch.tensor(next_states.copy(), dtype=torch.float32)  
            dones       = torch.tensor(dones, dtype=torch.float32)        
            
            # バッチ内のすべてのQ値を求める
            q_all = q_net(states)          
            # 各アクションと一致するQ値を選択する        
            q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  

            # ターゲット（Y_t）
            with torch.no_grad():
                q_next_all = target_net(next_states)      
                q_next_max, _ = q_next_all.max(dim=1)      
                targets = rewards + gamma * q_next_max * (1.0 - dones)

            # 誤差
            loss = nn.MSELoss()(q_sa, targets)
            print(loss)
            # 誤差逆伝播（Qネット）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())
# ---------------------------------------------------------------------------------

# ===============================
# 🎉 全エピソード完了 → モデル保存
# ===============================
torch.save(q_net.state_dict(), "qnet_final.pth")
torch.save(target_net.state_dict(), "target_final.pth")
print("モデルを保存しました。")
