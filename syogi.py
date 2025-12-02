import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class TicTacToe5x5:
    def __init__(self):
        self.board = np.array([
            [-5, -4, -3, -2, -1],  # 0-4 (プレイヤー2の駒)
            [0, 0, 0, 0, 0],       # 5-9
            [0, 0, 0, 0, 0],       # 10-14
            [0, 0, 0, 0, 0],       # 15-19
            [1, 2, 3, 4, 5]        # 20-24 (プレイヤー1の駒)
        ])
        self.current_player = 1
        self.winner = None
        self.fig = None
        self.ax = None
        self.is_gui_mode = False
        self.last_reward = 0
        self.done = False

    def display(self):
        """Display the board"""
        print(f"\nCurrent Player: {'O' if self.current_player == 1 else 'X'}")
        print("   0   1   2   3   4")
        print("  " + "---" * 5)
        for i in range(5):
            row_display = f"{i}|"
            for j in range(5):
                if np.abs(self.board[i][j]) == 0:
                    row_display += "   |"
                elif np.abs(self.board[i][j]) == 1:
                    row_display += " O |"
                elif np.abs(self.board[i][j]) == 2:
                    row_display += " X |"
                elif np.abs(self.board[i][j]) == 3:
                    row_display += " □ |"
                elif np.abs(self.board[i][j]) == 4:
                    row_display += " △ |"
                elif np.abs(self.board[i][j]) == 5:
                    row_display += " ★ |"
            print(row_display)
            print("  " + "---" * 5)

    def is_valid_move(self, x, y):
        """Check if the move is valid"""
        if x < 0 or x >= 5 or y < 0 or y >= 5:
            return False
        return self.board[y][x] == 0

    def make_move(self, x, y):
        """Make a move"""
        if not self.is_valid_move(x, y):
            return False

        self.board[y][x] = self.current_player

        # Check for winner
        if self.check_winner(x, y):
            self.winner = self.current_player
        # Check for draw
        elif np.all(self.board != 0):
            self.winner = 0
        else:
            # Switch player
            self.current_player = 3 - self.current_player

        return True

    def check_winner(self, x, y):
        """Check for winner (from last placed position)"""
        player = self.board[y][x]

        # Check horizontal
        if self.check_line(y, 0, 0, 1, player):
            return True

        # Check vertical
        if self.check_line(0, x, 1, 0, player):
            return True

        # Check diagonal (top-left to bottom-right)
        if self.check_diagonal(player, 1, 1):
            return True

        # Check diagonal (top-right to bottom-left)
        if self.check_diagonal(player, 1, -1):
            return True

        return False

    def check_line(self, start_y, start_x, dy, dx, player):
        """Check if 3 pieces are aligned in the specified direction"""
        count = 0
        y, x = start_y, start_x

        while 0 <= y < 5 and 0 <= x < 5:
            if self.board[y][x] == player:
                count += 1
                if count >= 3:
                    return True
            else:
                count = 0
            y += dy
            x += dx

        return False

    def check_diagonal(self, player, dy, dx):
        """Check if 3 pieces are aligned diagonally"""
        # Check all diagonal lines starting from top/top-right
        if dx == 1:  # Top-left to bottom-right
            # Start from top edge
            for start_x in range(5):
                if self.check_diagonal_line(0, start_x, dy, dx, player):
                    return True
            # Start from left edge (skip (0,0) as it's already checked)
            for start_y in range(1, 5):
                if self.check_diagonal_line(start_y, 0, dy, dx, player):
                    return True
        else:  # Top-right to bottom-left
            # Start from top edge
            for start_x in range(5):
                if self.check_diagonal_line(0, start_x, dy, dx, player):
                    return True
            # Start from right edge (skip (0,4) as it's already checked)
            for start_y in range(1, 5):
                if self.check_diagonal_line(start_y, 4, dy, dx, player):
                    return True

        return False

    def check_diagonal_line(self, start_y, start_x, dy, dx, player):
        """Check if 3 pieces are aligned on the specified diagonal line"""
        count = 0
        y, x = start_y, start_x

        while 0 <= y < 5 and 0 <= x < 5:
            if self.board[y][x] == player:
                count += 1
                if count >= 3:
                    return True
            else:
                count = 0
            y += dy
            x += dx

        return False

    def is_game_over(self):
        """Check if the game is over"""
        return self.winner is not None

    def reset(self):
        """Reset the game"""
        self.board = np.array([
            [-5, -4, -3, -2, -1],  # 0-4 (プレイヤー2の駒)
            [0, 0, 0, 0, 0],       # 5-9
            [0, 0, 0, 0, 0],       # 10-14
            [0, 0, 0, 0, 0],       # 15-19
            [1, 2, 3, 4, 5]        # 20-24 (プレイヤー1の駒)
        ])
        self.current_player = 1
        self.winner = None
        self.done = False
        self.last_reward = 0

        # Initialize GUI if not already done
        if not self.is_gui_mode:
            self.is_gui_mode = True
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.fig.canvas.manager.set_window_title('5x5 Tic-Tac-Toe')
            plt.ion()  # Interactive mode on

        if self.ax is not None:
            self._draw_board()
            plt.pause(0.01)

        return self.get_state()

    def get_state(self):
        """Get current state"""
        return self.board.copy()

    def step(self, a_t):
        board_ = self.board.flatten()
        board_ = board_.tolist()
        koma_num = int((a_t / 25)) + 1

        if koma_num not in board_:
            reward = -10
            done = False
            info = {'valid_move': False}
            if self.is_gui_mode:
              self._draw_board()
              plt.pause(0.01)
            return self.board, reward, done, info


        before = board_.index(koma_num)

        after = (a_t % 25)

        e = 1
        if(after-before == 5):
            e = 0
        if(after-before == -5):
            e = 0
        if(after-before == 1 or after-before == -4):
            if((before%5) != 4):
                e = 0
        if(after-before == -1 or after-before == -6):
            if((before%5) != 0):
                e = 0

        if(board_[before] <= 0):
            e = 1
        if(board_[after] > 0):
            e = 1

        reward = 0
        done = False

        if(e == 1):
            reward = -10
            done = False
            info = {'valid_move': False}
            if self.is_gui_mode:
              self._draw_board()
              plt.pause(0.01)
            return self.board, reward, done, info

        if(board_[after] == -3):
            done = True

        if(done == True):
            self.winner = self.current_player

        self.current_player = 3 - self.current_player

        if(self.winner == 1):
            reward = 10
        if(self.winner == 2):
            reward = -10

        self.done = done
        self.last_reward = reward

        info = {'valid_move': True, 'winner': self.winner}

        #make_move()にあたる
        board_[after] = board_[before]
        board_[before] = 0

        #self.board = 常に反転。
        board2 = np.array(board_)
        board3 = -1 * board2
        board3 = board3[::-1]
        self.board = board3.reshape(5, 5)

        #board3 = 常に正の向きをどうにかして返すように。

        if(self.current_player == 1):
            if self.is_gui_mode:
              self._draw_board()
              plt.pause(0.01)
            return board2.reshape(5, 5), reward, done, info
        if(self.current_player == 2):
            if self.is_gui_mode:
              self._draw_board()
              plt.pause(0.01)
            return board3.reshape(5, 5), reward, done, info

    def get_valid_actions(self):
        """Get list of valid actions"""
        valid_actions = []
        for y in range(5):
            for x in range(5):
                if self.board[y][x] == 0:
                    valid_actions.append(y * 5 + x)
        return valid_actions

    def step_random(self):
        """Make a random move (for opponent)
        Returns: (state, reward, done, info)
        """
        valid_actions = self.get_valid_actions()

        if len(valid_actions) == 0:
            # No valid moves (shouldn't happen in normal play)
            return self.get_state(), 0, True, {'valid_move': False}

        # Choose random action
        a_t = random.randint(0,124)

        # Execute the action
        return a_t



    def _draw_board(self):
        """Draw the board"""
        self.ax.clear()
        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(0, 5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()

        reversed_board = self.board.copy()

        if(self.current_player==2):
          reversed_board = -1 *reversed_board
          reversed_board = reversed_board[::-1].reshape(5, 5)
          reversed_board = reversed_board[:, ::-1].reshape(5, 5)

        # Draw grid lines
        for i in range(6):
            self.ax.plot([i, i], [0, 5], 'k-', linewidth=2)
            self.ax.plot([0, 5], [i, i], 'k-', linewidth=2)

        # Draw O and X
        for y in range(5):
            for x in range(5):
                # Show position numbers for reference
                self.ax.text(x + 0.5, y + 0.1, str(y * 5 + x),
                           ha='center', va='top', fontsize=8, color='gray')

                # 現在のマス目の値を取得
                cell_value = reversed_board[y][x]
                # 値の絶対値を取得（マークの種類を判別するため）
                abs_value = np.abs(cell_value)

                # 符号に基づいて色を決定
                # 正の値 (>= 0) なら 'red'、負の値 (< 0) なら 'blue'
                draw_color = 'red' if cell_value >= 0 else 'blue'

                if abs_value == 1:  # O (Circle)
                    # O は 'fill=False' (線のみ) なので、edgecolor に draw_color を適用
                    circle = patches.Circle((x + 0.5, y + 0.5), 0.3,
                                            fill=False, edgecolor=draw_color, linewidth=3)
                    self.ax.add_patch(circle)

                elif abs_value == 2:  # X (Cross)
                    # X は 'r-' (赤線) の部分を draw_color を使った f-string に変更
                    self.ax.plot([x + 0.2, x + 0.8], [y + 0.2, y + 0.8],
                                f'{draw_color[0]}-', linewidth=3) # draw_color[0] は 'r' または 'b'
                    self.ax.plot([x + 0.2, x + 0.8], [y + 0.8, y + 0.2],
                                f'{draw_color[0]}-', linewidth=3)

                elif abs_value == 3:  # ☆ (Star) - マーカーとして描画
                    # マーカーの edge と face の両方に draw_color を適用
                    self.ax.plot(x + 0.5, y + 0.5, marker='*', markersize=50,
                                markeredgecolor=draw_color, markerfacecolor=draw_color,
                                linestyle='', linewidth=0)

                elif abs_value == 4:  # △ (Upward Triangle) - マーカーとして描画
                    self.ax.plot(x + 0.5, y + 0.5, marker='^', markersize=40,
                                markeredgecolor=draw_color, markerfacecolor=draw_color,
                                linestyle='', linewidth=0)

                elif abs_value == 5:  # □ (Square) - マーカーとして描画
                    self.ax.plot(x + 0.5, y + 0.5, marker='s', markersize=40,
                                markeredgecolor=draw_color, markerfacecolor=draw_color,
                                linestyle='', linewidth=0)

        # Coordinate labels
        self.ax.set_xticks(np.arange(5) + 0.5)
        self.ax.set_yticks(np.arange(5) + 0.5)
        self.ax.set_xticklabels(range(5))
        self.ax.set_yticklabels(range(5))

        # Title
        if self.winner is not None:
            if self.winner == 0:
                title = "Draw! Press 'r' to restart"
            else:
                winner_mark = "O" if self.winner == 1 else "X"
                title = f"Player {winner_mark} Wins! Press 'r' to restart"
        else:
            current_mark = "O" if self.current_player == 1 else "X"
            title = f"Current Player: {current_mark} (Press 0-24 to place)"

        self.ax.set_title(title, fontsize=16, pad=20)
        self.ax.grid(False)

        self.fig.canvas.draw()




def get_player_action(env):
    """Get player action input
    Returns: action (0-24), 'restart', 'quit', or None (on error)
    """
    try:
        user_input = input(f"\nPlayer {'O' if env.current_player == 1 else 'X'}'s turn. Enter action (0-24), 'r' to restart, or 'q' to quit: ").strip()

        if user_input.lower() == 'q':
            return 'quit'

        if user_input.lower() == 'r':
            return 'restart'

        # Convert input to action
        a_t = int(user_input)

        if a_t < 0 or a_t > 24:
            print("Error: Action must be between 0 and 24")
            return None

        return a_t

    except ValueError:
        print("Error: Please enter a valid number (0-24)")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def play_ai_turn(env):
    """Execute AI's turn
    Returns: True if game continues, False if game ended
    """
    print("\nAI (Player X) is thinking...")
    import time
    time.sleep(0.5)  # Add delay for better UX

    a_t = env.step_random()
    state, reward, done, info = env.step(a_t)
    print(f"AI played action: (reward: {reward})")

    return not done


def display_game_result(env):
    """Display game result"""
    env.display()

    if env.winner == 0:
        print("\n" + "=" * 40)
        print("Draw!")
        print("=" * 40)
    else:
        print("\n" + "=" * 40)
        if env.winner == 1:
            print("You Win!")
        else:
            print("AI Wins!")
        print("=" * 40)


def ask_restart():
    """Ask if player wants to restart
    Returns: True to restart, False to quit
    """
    restart = input("Play again? (y/n): ").strip().lower()
    return restart == 'y'


if __name__ == "__main__":
    # Instantiate as env
    env = TicTacToe5x5()

    print("=" * 40)
    print("5x5 Tic-Tac-Toe vs Random AI")
    print("=" * 40)
    print("Get 3 in a row (horizontal, vertical, or diagonal) to win")
    print("Position mapping: row * 5 + col (e.g., position 7 = row 1, col 2)")
    print("You are Player O, AI is Player X")
    print("=" * 40)

    # Reset and initialize the board
    state = env.reset()

    # Main game loop
    try:
        while True:
            # Display current state
            env.display()
            print(f"Valid actions: {env.get_valid_actions()}")

            # AI's turn
            if env.current_player == 2 and not env.is_game_over():
                if not play_ai_turn(env):
                    # Game ended after AI move
                    display_game_result(env)
                    if ask_restart():
                        state = env.reset()
                    else:
                        break
                continue

            # Human player's turn
            action = get_player_action(env)

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
            state, reward, done, info = env.step(action)

            if not info['valid_move']:
                print("Invalid move! Please choose an empty square.")
                continue

            print(f"Reward: {reward}")

            # Check if game is over
            if done:
                display_game_result(env)
                if ask_restart():
                    state = env.reset()
                else:
                    break

    except KeyboardInterrupt:
        print("\nExiting game...")
    finally:
        plt.close('all')