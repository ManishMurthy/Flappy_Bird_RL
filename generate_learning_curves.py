import matplotlib.pyplot as plt
import numpy as np

# Create sample data (replace with your actual training data from 5000 episodes)
episodes = np.arange(1, 5001)
rewards = np.zeros(5000)

# Initial exploration phase
rewards[:200] = 2 + np.random.normal(0, 1, 200)

# Rapid improvement phase
x = np.arange(200, 1000)
rewards[200:1000] = 2 + 15 * (x - 200) / 800 + np.random.normal(0, 3, 800)

# Refinement phase
x = np.arange(1000, 5000)
rewards[1000:5000] = 17 + 3 * np.sin(x/50) + np.random.normal(0, 2, 4000)

# Smooth the data for better visualization
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

rewards_smooth = moving_average(rewards, 100)
episodes_smooth = episodes[50:-49]  # Adjust for the moving window

# Calculate epsilon values (exploration rate)
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
epsilon = np.zeros(5000)
eps = epsilon_start
for i in range(5000):
    epsilon[i] = eps
    eps = max(epsilon_min, eps * epsilon_decay)

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot episode rewards
ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
ax1.plot(episodes_smooth, rewards_smooth, linewidth=2, color='blue', label='Moving Avg')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Episode Reward', fontsize=12)
ax1.set_title('Learning Progress Over 5000 Episodes', fontsize=14)
ax1.grid(alpha=0.3)
ax1.legend()

# Plot epsilon values
ax2.plot(episodes, epsilon, color='red', linewidth=2)
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
ax2.set_title('Exploration Rate Decay', fontsize=14)
ax2.grid(alpha=0.3)

# Annotate learning phases
ax1.annotate('Initial Exploration', xy=(100, 3), xytext=(100, 5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
ax1.annotate('Rapid Improvement', xy=(500, 10), xytext=(500, 5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
ax1.annotate('Policy Refinement', xy=(2500, 18), xytext=(2500, 12),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
plt.savefig('figures/learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("Learning curves figure saved!")