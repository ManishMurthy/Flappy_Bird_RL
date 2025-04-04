import numpy as np
from flappy_bird_gym import FlappyBirdEnv
from dqn_agent_pytorch import DQNAgent

def test_agent(model_file):
    # Create environment
    env = FlappyBirdEnv()
    state_size = 5
    action_size = 2
    
    # Create agent and load trained weights
    agent = DQNAgent(state_size, action_size)
    agent.load(model_file)
    agent.epsilon = 0.01  # Set epsilon to a small value for some exploration
    
    # Run test episodes
    for i in range(5):
        state = env.reset()
        env.render_mode = True
        done = False
        score = 0
        
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            score = info['score']
            
        print(f"Test Episode {i+1}, Score: {score}")
    
    env.close()

if __name__ == "__main__":
    # Use your best model
    test_agent("flappy_bird_model_1000.pt")