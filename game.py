import matplotlib.pyplot as plt
from marubatu5 import TicTacToe5x5


def get_player_action():
    """Get player action from keyboard input
    Returns: action (0-24), 'restart', 'quit', or None (on error)
    """
    try:
        user_input = input("\nYour turn (Player O). Enter action (0-24), 'r' to restart, or 'q' to quit: ").strip()
        
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


def play_game():
    """Main game loop"""
    # Create environment
    env = TicTacToe5x5()
    
    print("=" * 50)
    print("5x5 Tic-Tac-Toe vs Random AI")
    print("=" * 50)
    print("Get 3 in a row to win!")
    print("You: Player O (Blue Circle)")
    print("AI:  Player X (Red Cross)")
    print("=" * 50)
    
    # Reset and initialize
    state = env.reset()
    
    # Main game loop
    try:
        while True:
            # Display current state
            env.display()
            # print(f"\nValid actions: {env.get_valid_actions()}")
            
            # Check if it's AI's turn
            if env.current_player == 2 and not env.is_game_over():
                print("\nAI (Player X) is thinking...")
                import time
                time.sleep(0.3)
                
                state, reward, done, info = env.step_random()
                print(f"AI played!")
                
                # Check if game ended
                if done:
                    env.display()
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
                        state = env.reset()
                    else:
                        break
                continue
            
            # Human player's turn
            action = get_player_action()
            
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
            
            # Check if game ended
            if done:
                env.display()
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
                    state = env.reset()
                else:
                    break
    
    except KeyboardInterrupt:
        print("\nExiting game...")
    finally:
        plt.close('all')
        print("Game closed.")

