import numpy as np
import matplotlib.pyplot as plt
import torch
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import time
import os

def train_dqn_agent(episodes=1000, use_assets=True):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = FlappyBirdEnv(use_assets=use_assets)
    
    # Get state and action dimensions
    state_size = 5  # Number of observations (bird_y, velocity, horizontal_distance, height_diff_top, height_diff_bottom)
    action_size = 2  # Number of actions (do nothing, flap)
    
    # Create agent
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Training parameters
    batch_size = 32
    target_update_frequency = 10  # How often to update target network
    
    # Create folder for saving models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Track scores and epsilon values for plotting
    scores = []
    epsilons = []
    
    # Start training
    for e in range(episodes):
        # Reset environment
        state = env.reset()
        
        total_reward = 0
        max_steps = 10000  # Limit on steps per episode
        
        for step in range(max_steps):
            # For visualization (can be turned off for faster training)
            if e % 50 == 0:  # Render every 50 episodes
                env.render()
                
            # Choose action
            action = agent.act(state)
            
            # Take action and observe result
            next_state, reward, done, info = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Accumulate reward
            total_reward += reward
            
            # Train the agent (experience replay)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # If game over, exit loop
            if done:
                # Update target model periodically
                if e % target_update_frequency == 0:
                    agent.update_target_model()
                    
                break
        
        # Save scores and epsilon values
        scores.append(total_reward)
        epsilons.append(agent.epsilon)
        
        # Print progress
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        print(f"Episode: {e+1}/{episodes}, Score: {info['score']}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Avg Score (last 100): {avg_score:.2f}")
        
        # Save model every 100 episodes
        if (e+1) % 100 == 0:
            agent.save(f"models/flappy_dqn_episode_{e+1}.pt")
    
    # Save final model
    agent.save("models/flappy_dqn_final.pt")
    env.close()
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    return agent

if __name__ == "__main__":
    train_dqn_agent()