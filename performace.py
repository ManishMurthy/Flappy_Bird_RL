import matplotlib.pyplot as plt
import numpy as np

# Example dummy data – replace this with your actual recorded values
# For example: average reward per episode
# reward_history = [r1, r2, r3, ..., r1000]
reward_history = np.loadtxt("rewards.txt")  # If stored in a .txt or .csv file
episodes = np.arange(1, len(reward_history) + 1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(episodes, reward_history, label='Average Reward per Episode', color='royalblue')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress: Reward Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('training_progress_reward.png')
plt.show()