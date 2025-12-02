"""
PlayGame.py
学習済みDQNエージェントと対戦するプログラム
"""

import numpy as np
from syogi import TicTacToe5x5
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn


# DQN用のニューラルネットワーク (dqn_game.pyと同じ構造)
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25, 128)  # 5x5=25の状態
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 25)  # 25の行動空間 (5x5の盤面)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # Q(s, ·)


class AI:
    """
    学習済みDQNモデルを使用してゲームをプレイするAIクラス
    """
    
    def __init__(self, model_path):
        """
        初期化: 学習済みモデルを読み込む
        
        Args:
            model_path: 学習済みモデルファイルのパス
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork().to(self.device)
        
        
        # 学習済みモデルを読み込む
        try:
            self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.q_net.eval()  # 推論モードに設定
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def select_action(self, state, valid_actions):
        """
        最善手を選択する
        
        Args:
            state: 現在の状態 (numpy配列, shape=(25,))
            valid_actions: 有効なアクションのリスト
            
        Returns:
            action: 選択されたアクション
        """
        with torch.no_grad():
            # 状態をTensorに変換
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Q値を計算
            q_values = self.q_net(state_tensor).squeeze(0)
            
            # CPUに移動してnumpy配列に変換
            q_values = q_values.cpu().numpy()
            
            # 無効なアクションのQ値を-infに設定
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            
            # 最大Q値を持つアクションを選択
            action = np.argmax(masked_q_values)
            
            return action


class InteractiveGame:
    """
    マウスクリックで操作できる対話型ゲーム
    """
    
    def __init__(self, agent, mode='exchange'):
        """
        初期化
        
        Args:
            agent: 学習済みDQNエージェント
            mode: ゲームモード ('exchange': 交代プレイ, 'ai': AI対戦)
        """
        self.agent = agent
        self.env = TicTacToe5x5()
        self.player_action = None
        self.waiting_for_player = False
        self.mode = mode  # ゲームモード
        
        # 将棋用の2クリック操作用の状態
        self.selecting_piece = True  # True: 駒選び, False: 移動先選び
        self.selected_piece_pos = None  # 選択された駒の位置 (x, y)
        
    def is_valid_piece_move(self, from_x, from_y, to_x, to_y):
        """
        駒が指定位置に移動可能かチェック（全ての駒は金の動き）
        
        Args:
            from_x, from_y: 移動元の座標
            to_x, to_y: 移動先の座標
            
        Returns:
            bool: 移動可能ならTrue
        """
        # 座標範囲チェック
        if not (0 <= to_x < 5 and 0 <= to_y < 5):
            return False
        
        # 移動先が空きマスかチェック
        if self.env.board[to_y][to_x] != 0:
            return False
        
        piece = self.env.board[from_y][from_x]
        
        # 移動量を計算
        dx = to_x - from_x
        dy = to_y - from_y
        
        # 全ての駒は金の動き: 前、左右、斜め前、真後ろに1マス
        # プレイヤー1(正の値): 上方向が前、プレイヤー2(負の値): 下方向が前
        if abs(piece) > 0:  # プレイヤー1
            # 前(上): dx=0, dy=-1
            # 左右: dx=±1, dy=0
            # 斜め前: dx=±1, dy=-1
            # 真後ろ: dx=0, dy=1
            valid_moves = [
                (0, -1),   # 前
                (-1, 0),   # 左
                (1, 0),    # 右
                (-1, -1),  # 斜め前左
                (1, -1),   # 斜め前右
                (0, 1)     # 真後ろ
            ]
        else:  # プレイヤー2
            # 前(下): dx=0, dy=1
            # 左右: dx=±1, dy=0
            # 斜め前: dx=±1, dy=1
            # 真後ろ: dx=0, dy=-1
            valid_moves = [
                (0, 1),    # 前
                (-1, 0),   # 左
                (1, 0),    # 右
                (-1, 1),   # 斜め前左
                (1, 1),    # 斜め前右
                (0, -1)    # 真後ろ
            ]
        
        return (dx, dy) in valid_moves
    
    def on_click(self, event):
        """
        マウスクリックイベント処理（将棋用2クリック操作）
        
        Args:
            event: クリックイベント
        """
        # プレイヤーのターンでない場合は無視
        if not self.waiting_for_player:
            return
            
        # クリック位置が盤面内かチェック
        if event.xdata is None or event.ydata is None:
            return
            
        # クリック位置を座標に変換
        x = int(event.xdata)
        y = int(event.ydata)
        
        # 座標が範囲内かチェック
        if not (0 <= x < 5 and 0 <= y < 5):
            print(f"クリック位置が盤面外です: ({x}, {y})")
            return
        
        # 状態に応じた処理
        if self.selecting_piece:
            # 駒選び: 自分の駒があるかチェック
            piece = self.env.board[y][x]
            player_value = self.env.current_player
            
            # プレイヤー1は正の値、プレイヤー2は負の値の駒を持つ
            if player_value == 1 and piece > 0:
                # プレイヤー1の駒を選択
                self.selected_piece_pos = (x, y)
                self.selecting_piece = False
                print(f"駒を選択しました: 位置({x}, {y}), 種類={abs(piece)}")
            elif player_value == 2 and piece < 0:
                # プレイヤー2の駒を選択
                self.selected_piece_pos = (x, y)
                self.selecting_piece = False
                print(f"駒を選択しました: 位置({x}, {y}), 種類={abs(piece)}")
            else:
                print(f"エラー: 自分の駒がありません。位置({x}, {y})")
        
        else:
            # 移動先選び: 選択された駒が移動可能かチェック
            from_x, from_y = self.selected_piece_pos
            
            if self.is_valid_piece_move(from_x, from_y, x, y):
                # アクションを計算: (駒番号-1) * 25 + 移動先のインデックス
                piece = self.env.board[from_y][from_x]
                koma_num = abs(piece)  # 駒の番号 (1-5)
                to_index = y * 5 + x   # 移動先のインデックス (0-24)
                self.player_action = (koma_num - 1) * 25 + to_index
                
                self.waiting_for_player = False
                self.selecting_piece = True  # 次回は駒選びに戻る
                self.selected_piece_pos = None
                print(f"移動先を選択しました: ({from_x}, {from_y}) -> ({x}, {y})")
                print(f"アクション: 駒番号={koma_num}, 移動先インデックス={to_index}, action={self.player_action}")
            else:
                print(f"エラー: その位置には移動できません。位置({x}, {y})")
                # エラーの場合は駒選びに戻る
                self.selecting_piece = True
                self.selected_piece_pos = None
                print("駒の選択からやり直してください。")
    
    def _exchange_on_click(self, event):
        """
        交代プレイモード用のクリックイベント処理
        プレイヤー1とプレイヤー2が交互に操作する
        
        Args:
            event: クリックイベント
        """
        # 現在のプレイヤーを保存
        current_player = self.env.current_player
        
        if current_player == 1:
            # プレイヤー1（自分）のターン: 通常のon_clickを実行
            self.on_click(event)
        else:
            # プレイヤー2（相手）のターン: on_clickを実行後、座標を反転
            # まず通常通りon_clickを実行
            self.on_click(event)
            
            # player_actionが計算された場合、to_indexを盤面反転に対応させる
            if self.player_action is not None and not self.waiting_for_player:
                # アクションを分解: (駒番号-1) * 25 + to_index
                koma_num = (self.player_action // 25) + 1
                to_index = self.player_action % 25
                
                # to_indexを座標に変換
                to_x = to_index % 5
                to_y = to_index // 5
                
                # 盤面を反転（上下反転）
                flipped_to_y = 4 - to_y
                flipped_to_x = 4 - to_x
                
                # 反転後のto_indexを計算
                flipped_to_index = flipped_to_y * 5 + flipped_to_x
                
                # アクションを再計算
                self.player_action = (koma_num - 1) * 25 + flipped_to_index
                
                print(f"座標反転: ({to_x}, {to_y}) -> ({flipped_to_x}, {flipped_to_y})")
                print(f"反転後アクション: {self.player_action}")
                
    def get_player_action(self):
        """
        プレイヤーのアクションを取得（クリック待ち）
        
        Returns:
            action: プレイヤーが選択したアクション ((from_x, from_y), (to_x, to_y))のタプル
        """
        self.player_action = None
        self.waiting_for_player = True
        self.selecting_piece = True  # 駒選びから開始
        self.selected_piece_pos = None
        
        player_name = "Player 1 (下側)" if self.env.current_player == 1 else "Player 2 (上側)"
        print(f"\n{player_name}のターンです。")
        print("1. まず自分の駒をクリックしてください")
        print("2. 次に移動先をクリックしてください")
        
        # クリックを待つ（2クリック完了まで）
        while self.waiting_for_player:
            plt.pause(0.1)
            
        return self.player_action
    
        
    def ai_turn(self):
        """
        AIのターンを実行
        
        Returns:
            done: ゲームが終了した場合True
        """
        print("\nAI (Player O) is thinking...")
        time.sleep(0.5)
        
        state = self.env.get_state()
        valid_actions = self.env.get_valid_actions()
        
        # AIがアクションを選択
        action = self.agent.select_action(state.flatten(), valid_actions)
        
        # アクションを実行
        state, reward, done, info = self.env.step(action)
        
        print(f"AI played action: {action}")
        print(f"Position: (x={action % 5}, y={action // 5})")
        
        return done
        
    def display_result(self):
        """ゲーム結果を表示"""
        print("\n" + "=" * 50)
        if self.env.winner == 0:
            print("Draw!")
        elif self.env.winner == 1:
            print("Player 1 (下側) Wins!")
        else:
            print("Player 2 (上側) Wins!")
        print("=" * 50)
        
    def play(self):
        """ゲームをプレイ"""
        print("=" * 50)
        if self.mode == 'exchange':
            print("5x5 将棋風ゲーム (交代プレイモード)")
            print("=" * 50)
            print("相手の王(★)を取ると勝利")
            print("操作方法: 駒をクリック → 移動先をクリック")
            print("プレイヤー1(下側)とプレイヤー2(上側)が交互に操作")
        else:
            print("5x5 将棋風ゲーム (AI対戦モード)")
            print("=" * 50)
            print("相手の王(★)を取ると勝利")
            print("操作方法: 駒をクリック → 移動先をクリック")
            print("あなた: プレイヤー1(下側), AI: プレイヤー2(上側)")
        print("Close the window to quit")
        print("=" * 50)
        
        # ゲームをリセット（GUIを初期化）
        state = self.env.reset()
        
        # モードに応じてクリックイベントを登録
        if self.mode == 'exchange':
            self.env.fig.canvas.mpl_connect('button_press_event', self._exchange_on_click)
        else:
            self.env.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        try:
            # ゲームループ
            while not self.env.is_game_over():
                if self.mode == 'exchange':
                    # 交代プレイモード: 両プレイヤーが手動で操作
                    action = self.get_player_action()
                else:
                    # AI対戦モード
                    if self.env.current_player == 1:
                        # プレイヤー1のターン
                        action = self.get_player_action()
                    else:
                        # プレイヤー2（AI）のターン
                        print("\nAI (Player 2) is thinking...")
                        time.sleep(0.5)
                        state = self.env.get_state()
                        valid_actions = self.env.get_valid_actions()
                        action = self.agent.select_action(state.flatten(), valid_actions)
                        print(f"AI played action: {action}")
                
                if action is None:
                    continue
                
                # アクションを実行
                state, reward, done, info = self.env.step(action)
                
                if done:
                    break
            
            # 結果を表示
            self.display_result()
            
            # 結果を表示したまま待機
            print("\nClose the window to exit.")
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close('all')


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
            mode = 'exchange'
            print("\n交代プレイモードを選択しました\n")
            break
        elif mode_input == '2':
            mode = 'ai'
            print("\nAI対戦モードを選択しました\n")
            break
        else:
            print("無効な入力です。1 または 2 を入力してください。")
    
    # モデルパス
    MODEL_PATH = 'qnet_final.pth'  # dqn_game.pyで保存されたモデルファイル
    
    # エージェントの作成
    print("Loading trained DQN agent...")
    agent = AI(MODEL_PATH)    
    
    game = InteractiveGame(agent, mode=mode)
    game.play()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
