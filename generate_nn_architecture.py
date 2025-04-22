import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create figure for neural network architecture
fig, ax = plt.subplots(figsize=(12, 6))

# Layer positions and sizes
layer_positions = [1, 3, 5, 7, 9]
layer_sizes = [5, 64, 64, 32, 2]  # Input, hidden layers, output
layer_colors = ['#FFC107', '#4CAF50', '#4CAF50', '#4CAF50', '#2196F3']
layer_labels = ['Input\n(State)', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Hidden\nLayer 3', 'Output\n(Q-values)']

# Draw each layer
for i, (pos, size, color, label) in enumerate(zip(layer_positions, layer_sizes, layer_colors, layer_labels)):
    # Scale down the larger layers for better visualization
    display_size = min(size, 20) if size > 20 else size
    
    # Draw neurons as circles
    y_positions = np.linspace(0, 4, display_size)
    for y_pos in y_positions:
        circle = plt.Circle((pos, y_pos), 0.2, color=color, fill=True)
        ax.add_patch(circle)
    
    # If there are more neurons than displayed, add dots
    if size > display_size:
        ax.text(pos, 4.5, f"({size} neurons)", ha='center', va='center', fontsize=10)
    
    # Add layer label
    ax.text(pos, -1, label, ha='center', va='center', fontsize=12)

# Add connections between layers
for i in range(len(layer_positions)-1):
    pos1, pos2 = layer_positions[i], layer_positions[i+1]
    size1, size2 = min(layer_sizes[i], 20), min(layer_sizes[i+1], 20)
    
    y1_positions = np.linspace(0, 4, size1)
    y2_positions = np.linspace(0, 4, size2)
    
    # Draw a sample of connections (not all to avoid clutter)
    for _ in range(min(size1, size2) * 2):
        start_idx = np.random.randint(0, len(y1_positions))
        end_idx = np.random.randint(0, len(y2_positions))
        
        ax.plot([pos1, pos2], [y1_positions[start_idx], y2_positions[end_idx]], 
                color='gray', alpha=0.3, linewidth=0.5)

# Add activation function labels
activation_positions = [(pos1 + pos2) / 2 for pos1, pos2 in zip(layer_positions[:-1], layer_positions[1:])]
activation_labels = ['ReLU', 'ReLU', 'ReLU', 'Linear']

for pos, label in zip(activation_positions, activation_labels):
    ax.text(pos, 5, label, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", fc='#E3F2FD', ec="black", alpha=0.7))

# Set plot limits and remove axes
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 6)
ax.axis('off')

plt.title('Deep Q-Network Architecture for Flappy Bird', fontsize=14)
plt.tight_layout()
plt.savefig('figures/dqn_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Neural network architecture figure saved!")