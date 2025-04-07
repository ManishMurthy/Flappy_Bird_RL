import numpy as np
import time
import matplotlib.pyplot as plt
from flappy_bird_gym import FlappyBirdEnv
from dqn_agent_pytorch import DQNAgent

def test_agent(model_file, num_episodes=10, render=True):
    """
    Test a trained DQN agent and collect performance metrics
    
    Args:
        model_file: Path to the trained model file
        num_episodes: Number of episodes to test
        render: Whether to render the game visually
    """
    # Create environment
    env = FlappyBirdEnv()
    state_size = 3
    action_size = 2
    
    # Create agent and load trained weights
    agent = DQNAgent(state_size, action_size)
    agent.load(model_file)
    agent.epsilon = 0.01  # Set epsilon to a small value for some exploration
    
    # Metrics to track
    scores = []
    episode_lengths = []
    pipe_counts = []
    actions_taken = []
    rewards_per_episode = []
    
    # Also track a baseline for comparison (random agent)
    baseline_scores = []
    
    # Run test episodes for trained agent
    print(f"\n--- Testing trained agent from {model_file} ---")
    for i in range(num_episodes):
        state = env.reset()
        env.render_mode = render
        
        done = False
        step_count = 0
        episode_score = 0
        total_reward = 0
        pipes_cleared = 0
        flap_count = 0
        
        start_time = time.time()
        
        while not done:
            # Get action from agent
            action = agent.act(state)
            actions_taken.append(action)
            
            if action == 1:  # If flap action
                flap_count += 1
                
            # Take step
            state, reward, done, info = env.step(action)
            
            # Update metrics
            step_count += 1
            total_reward += reward
            
            # Get current score using get() with default value
            current_score = info.get('score', 0)
            if current_score > episode_score:
                pipes_cleared += 1
                episode_score = current_score
                
        # Record episode results
        elapsed_time = time.time() - start_time
        scores.append(episode_score)
        episode_lengths.append(step_count)
        pipe_counts.append(pipes_cleared)
        rewards_per_episode.append(total_reward)
        
        # Calculate flap percentage
        flap_percentage = (flap_count / step_count) * 100 if step_count > 0 else 0
        
        print(f"Episode {i+1}: Score={episode_score}, Steps={step_count}, " 
              f"Reward={total_reward:.2f}, Pipes cleared={pipes_cleared}, "
              f"Flap %={flap_percentage:.1f}%, Time={elapsed_time:.2f}s")
    
    # Run a few baseline episodes with random actions for comparison
    print("\n--- Testing random agent (baseline) ---")
    for i in range(3):  # Just 3 episodes for baseline
        state = env.reset()
        env.render_mode = render
        
        done = False
        baseline_score = 0
        
        while not done:
            # Random action
            action = np.random.randint(0, action_size)
            state, reward, done, info = env.step(action)
            baseline_score = info.get('score', 0)
            
        baseline_scores.append(baseline_score)
        print(f"Baseline Episode {i+1}: Score={baseline_score}")
    
    # Print summary statistics
    print("\n--- Test Results Summary ---")
    print(f"Trained Agent (over {num_episodes} episodes):")
    print(f"  Average score: {np.mean(scores):.2f}")
    print(f"  Max score: {max(scores)}")
    print(f"  Min score: {min(scores)}")
    print(f"  Average episode length: {np.mean(episode_lengths):.2f} steps")
    print(f"  Average pipes cleared: {np.mean(pipe_counts):.2f}")
    print(f"  Average reward: {np.mean(rewards_per_episode):.2f}")
    print(f"  Flap percentage: {(actions_taken.count(1) / len(actions_taken) * 100):.2f}%")
    
    print("\nBaseline (random actions):")
    print(f"  Average score: {np.mean(baseline_scores):.2f}")
    print(f"  Max score: {max(baseline_scores)}")
    
    # Plot results if there are multiple episodes
    if num_episodes > 1:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, num_episodes+1), scores)
        plt.axhline(y=np.mean(scores), color='r', linestyle='-', label=f'Avg: {np.mean(scores):.1f}')
        plt.axhline(y=np.mean(baseline_scores), color='g', linestyle='--', label=f'Random: {np.mean(baseline_scores):.1f}')
        plt.title('Scores per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.bar(range(1, num_episodes+1), episode_lengths)
        plt.axhline(y=np.mean(episode_lengths), color='r', linestyle='-', label=f'Avg: {np.mean(episode_lengths):.1f}')
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('test_results.png')
        plt.show()
    
    env.close()
    return scores

if __name__ == "__main__":
    # Use your best model
    test_agent("models/flappy_bird_model_1000.pt", num_episodes=10)  # Change to your actual model file