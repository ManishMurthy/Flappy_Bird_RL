import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create figure for experience replay mechanism
fig, ax = plt.subplots(figsize=(12, 6))

# Memory buffer as a queue
buffer_x = 2
buffer_y = 5
buffer_width = 8
buffer_height = 2

# Draw the replay buffer
buffer = Rectangle((buffer_x, buffer_y), buffer_width, buffer_height, 
                  facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
ax.add_patch(buffer)
ax.text(buffer_x + buffer_width/2, buffer_y + buffer_height + 0.3, 
        'Experience Replay Buffer', ha='center', fontsize=14)

# Draw memory entries
n_entries = 8
entry_width = buffer_width / n_entries
for i in range(n_entries):
    x = buffer_x + i * entry_width
    entry = Rectangle((x, buffer_y), entry_width, buffer_height, 
                     facecolor='white', edgecolor='#1976D2', linewidth=1)
    ax.add_patch(entry)
    ax.text(x + entry_width/2, buffer_y + buffer_height/2, f'(s,a,r,s\')', 
            ha='center', va='center', fontsize=8)

# Draw newest and oldest markers
ax.text(buffer_x + entry_width/2, buffer_y - 0.3, 'Oldest', ha='center', fontsize=10)
ax.text(buffer_x + buffer_width - entry_width/2, buffer_y - 0.3, 'Newest', ha='center', fontsize=10)

# Draw the agent interaction
agent_x = 6
agent_y = 2
environment_x = 10
environment_y = 2

# Draw the agent and environment
agent_circle = plt.Circle((agent_x, agent_y), 0.8, facecolor='#4CAF50', edgecolor='black')
ax.add_patch(agent_circle)
ax.text(agent_x, agent_y, 'Agent', ha='center', va='center', fontsize=12)

env_rect = Rectangle((environment_x - 1.5, environment_y - 0.8), 3, 1.6, 
                    facecolor='#FFC107', edgecolor='black')
ax.add_patch(env_rect)
ax.text(environment_x, environment_y, 'Environment', ha='center', va='center', fontsize=12)

# Draw the interaction arrows
# Agent to environment (action)
action_arrow = FancyArrowPatch((agent_x + 0.8, agent_y), (environment_x - 1.5, environment_y),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
ax.add_patch(action_arrow)
ax.text(agent_x + 1.5, agent_y + 0.3, 'action (a)', color='blue', fontsize=10)

# Environment to agent (state, reward)
reward_arrow = FancyArrowPatch((environment_x - 1.5, environment_y - 0.4), 
                              (agent_x + 0.8, agent_y - 0.4),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
ax.add_patch(reward_arrow)
ax.text(agent_x + 1.5, agent_y - 0.7, 'state (s), reward (r)', color='red', fontsize=10)

# New experience to buffer
new_exp_arrow = FancyArrowPatch((agent_x, agent_y + 0.8), (buffer_x + 7 * entry_width, buffer_y),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
ax.add_patch(new_exp_arrow)
ax.text(agent_x + 2, agent_y + 1.2, 'Store new experience', color='green', fontsize=10)

# Sample from buffer to training
sample_arrow = FancyArrowPatch((buffer_x + 3 * entry_width, buffer_y), (agent_x, agent_y + 0.8),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='purple')
ax.add_patch(sample_arrow)
ax.text(agent_x - 2, agent_y + 1.2, 'Sample random batch', color='purple', fontsize=10)

# Set plot limits and remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

plt.title('Experience Replay Mechanism', fontsize=16)
plt.tight_layout()
plt.savefig('figures/experience_replay.png', dpi=300, bbox_inches='tight')
plt.close()

print("Experience replay mechanism figure saved!")