import matplotlib.pyplot as plt
import numpy as np

# Create a grid of bird positions and pipe heights
bird_positions = np.linspace(0, 1, 100)  # Normalized bird y-position
pipe_positions = np.linspace(0.2, 0.8, 100)  # Normalized pipe height
X, Y = np.meshgrid(bird_positions, pipe_positions)

# Create decision boundary data
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        gap_center = Y[i, j]
        bird_y = X[i, j]
        
        # Simple decision logic (replace with your model's prediction if possible)
        if bird_y < gap_center - 0.1:  # Bird is below pipe gap
            Z[i, j] = 1  # Flap
        elif bird_y > gap_center + 0.1:  # Bird is above pipe gap
            Z[i, j] = 0  # Don't flap
        else:
            # In the gap, behavior depends on velocity (simulated here)
            velocity_factor = 0.5 * np.sin(bird_y * 10)  # Simulated velocity influence
            Z[i, j] = 1 if velocity_factor > 0 else 0

# Create a figure with a cleaner design
plt.figure(figsize=(10, 8))
cmap = plt.cm.coolwarm
plt.contourf(X, Y, Z, cmap=cmap, alpha=0.8, levels=[-0.5, 0.5, 1.5])

# Add contour lines for clarity
contour = plt.contour(X, Y, Z, colors='black', linewidths=0.5, levels=[0.5])

# Add pipe gap visualization
plt.axhspan(0.4, 0.6, xmin=0.3, xmax=0.4, color='green', alpha=0.3)  # Example pipe gap

# Add bird at different positions
plt.scatter([0.2, 0.35, 0.5], [0.3, 0.5, 0.7], color='yellow', s=100, edgecolor='black', zorder=5)

# Add title and labels
plt.title('Agent Decision Boundary', fontsize=16)
plt.xlabel('Bird Vertical Position (Normalized)', fontsize=12)
plt.ylabel('Pipe Gap Center (Normalized)', fontsize=12)

# Add a colorbar
cbar = plt.colorbar(ticks=[0, 1])
cbar.set_label('Action', fontsize=12)
cbar.ax.set_yticklabels(['Do Nothing', 'Flap'])

plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('figures/decision_boundary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Decision boundary figure saved!")