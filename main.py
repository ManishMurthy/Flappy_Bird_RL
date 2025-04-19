import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Flappy Bird DQN')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--model', type=str, default='models/flappy_dqn_final.pt', help='model path for testing')
    parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to train')
    parser.add_argument('--render_freq', type=int, default=50, help='rendering frequency during training')
    parser.add_argument('--no-assets', action='store_true', help='disable game assets and use simple graphics')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Check if assets directory exists
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created 'assets' directory. Please add game assets to this folder.")
        print("See the game_assets_guide.md file for instructions.")
    
    # Determine whether to use assets
    use_assets = not args.no_assets
    
    if args.mode == 'train':
        from train import train_dqn_agent
        print(f"Starting training for {args.episodes} episodes...")
        print(f"Using {'simple graphics' if not use_assets else 'game assets'}")
        agent = train_dqn_agent(episodes=args.episodes, use_assets=use_assets)
        print("Training completed!")
        
    elif args.mode == 'test':
        from src.test import test_trained_agent
        if os.path.exists(args.model):
            print(f"Testing agent with model: {args.model}")
            print(f"Using {'simple graphics' if not use_assets else 'game assets'}")
            test_trained_agent(args.model, use_assets=use_assets)
        else:
            print(f"Model {args.model} not found. Please train first or specify correct model path.")
            
    else:
        print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()