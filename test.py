import numpy as np
import torch
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pygame
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent

def test_trained_agent(model_path='models/flappy_dqn_final.pt', use_assets=True, num_episodes=50):
    # Create directory for screenshots if it doesn't exist
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    
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
    
    # Track scores
    all_scores = []
    best_score = 0
    
    # Test for specified number of episodes
    for e in range(num_episodes):
        state = env.reset()
        episode_score = 0
        
        done = False
        while not done:
            # Render the game
            env.render()
            time.sleep(0.03)  # Slow down for visibility
            
            # Choose action
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state and score
            state = next_state
            episode_score = info['score']
            
            # Take screenshot if this is a good score
            if (episode_score >= 10 and episode_score % 5 == 0) or (done and episode_score >= 10):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(screenshot_dir, f"flappy_score_{episode_score}_ep{e+1}_{timestamp}.png")
                
                # Take screenshot using pygame
                if hasattr(env, 'screen'):
                    pygame.image.save(env.screen, screenshot_path)
                    print(f"Screenshot saved: {screenshot_path}")
            
            if done:
                all_scores.append(episode_score)
                print(f"Episode: {e+1}, Score: {episode_score}")
                
                # Update best score and take screenshot if it's the best
                if episode_score > best_score:
                    best_score = episode_score
                    best_screenshot_path = os.path.join(screenshot_dir, f"flappy_best_score_{best_score}.png")
                    
                    # Take screenshot using pygame
                    if hasattr(env, 'screen'):
                        pygame.image.save(env.screen, best_screenshot_path)
                        print(f"New best score: {best_score}, screenshot saved!")
                
                time.sleep(1)  # Pause between episodes
                break
    
    # Display score statistics
    print("\nTest Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(all_scores):.2f}")
    print(f"Max Score: {np.max(all_scores)}")
    print(f"Min Score: {np.min(all_scores)}")
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=max(10, int(max(all_scores)/5)), alpha=0.7)
    plt.title('Flappy Bird Agent Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    score_dist_path = os.path.join(screenshot_dir, "score_distribution.png")
    plt.savefig(score_dist_path)
    print(f"Score distribution saved to: {score_dist_path}")
    
    env.close()

if __name__ == "__main__":
    test_trained_agent(num_episodes=100)