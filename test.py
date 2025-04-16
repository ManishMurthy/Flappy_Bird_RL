import numpy as np
import torch
import time
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent

def test_trained_agent(model_path='models/flappy_dqn_final.pt', use_assets=True):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = FlappyBirdEnv(use_assets=use_assets)
    
    # Get state and action dimensions
    state_size = 5  # Number of observations
    action_size = 2  # Number of actions
    
    # Create agent
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Load the trained model
    agent.load(model_path)
    
    # Set epsilon to minimum for mostly exploitation
    agent.epsilon = agent.epsilon_min
    
    # Test for 5 episodes
    for e in range(5):
        state = env.reset()
        
        done = False
        while not done:
            # Render
            env.render()
            time.sleep(0.03)  # Slow down for visibility
            
            # Choose action
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            
            if done:
                print(f"Episode: {e+1}, Score: {info['score']}")
                time.sleep(1)  # Pause between episodes
                break
    
    env.close()

if __name__ == "__main__":
    test_trained_agent()