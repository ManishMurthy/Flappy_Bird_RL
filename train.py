import numpy as np
from flappy_bird_gym import FlappyBirdEnv
from dqn_agent_pytorch import DQNAgent
import matplotlib.pyplot as plt
import os

def train_flappy_bird():
    # Create environment
    env = FlappyBirdEnv()
    state_size = 3  # size of our state vector
    action_size = 2  # flap or do nothing
    batch_size = 32
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    n_episodes = 1000
    max_steps = 10000
    scores = []
    
    # Make sure we have a models directory
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training loop
    for e in range(n_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        
        # Set render mode for visualization (only every 50 episodes to speed up training)
        if e % 50 == 0:
            env.render_mode = True
        else:
            env.render_mode = False
            
        for step in range(max_steps):
            # Agent selects action
            action = agent.act(state)
        
            # Environment takes action and returns new state and reward
            next_state, reward, done, info = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{n_episodes}, Score: {info.get('score', 0)}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                scores.append(info.get('score', 0))
                print(f"Info dictionary keys: {info.keys()}")
                break
                
            # Train the agent with batches of experiences
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
        # Save model every 100 episodes
        if (e + 1) % 100 == 0:
            agent.save(f"models/flappy_bird_model_{e+1}.pt")
            
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title('Flappy Bird Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('training_progress.png')
    plt.show()
    
    # Close environment
    env.close()
    
    return agent

if __name__ == "__main__":
    trained_agent = train_flappy_bird()