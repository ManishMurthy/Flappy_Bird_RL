import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

# Create Figures directory if it doesn't exist
os.makedirs('Figures', exist_ok=True)

# 1. PHYSICS SIMULATION VISUALIZATION
plt.figure(figsize=(10, 8))

# Set up the visualization area
plt.ylim(512, 0)  # Invert y-axis to match game coordinates
plt.xlim(0, 288)
plt.fill_between([0, 288], [0, 0], [512, 512], color='skyblue', alpha=0.3)
plt.fill_between([0, 288], [450, 450], [512, 512], color='tan')

# Simulate bird trajectories with different actions
# No flap trajectory (just falling with gravity)
gravity = 0.5
initial_y = 150
initial_velocity = 0
no_flap_positions = []
y = initial_y
velocity = initial_velocity

for x in range(0, 250, 5):
    no_flap_positions.append((x, y))
    velocity += gravity
    y += velocity

# Periodic flapping trajectory
flap_positions = []
y = initial_y
velocity = initial_velocity
for x in range(0, 250, 5):
    flap_positions.append((x, y))
    # Flap every 20 pixels
    if x % 40 == 0:
        velocity = -8  # Flap power
    else:
        velocity += gravity
    y += velocity
    
# Optimal flapping trajectory (to navigate through pipes)
pipe_x = 150
gap_center = 250
optimal_positions = []
y = initial_y
velocity = initial_velocity
for x in range(0, 250, 5):
    optimal_positions.append((x, y))
    # Smart flapping based on position and velocity
    if x < pipe_x - 50 and (y > gap_center or velocity < -2):
        # Do nothing, let gravity bring bird down
        velocity += gravity
    elif x < pipe_x and (y < gap_center or velocity > 2):
        # Flap to gain height
        velocity = -8  # Flap power
    else:
        velocity += gravity
    y += velocity

# Plot the trajectories
plt.plot([p[0] for p in no_flap_positions], [p[1] for p in no_flap_positions], 
         'r-', linewidth=2, label='No Flap (Gravity Only)')
plt.plot([p[0] for p in flap_positions], [p[1] for p in flap_positions], 
         'b-', linewidth=2, label='Periodic Flapping')
plt.plot([p[0] for p in optimal_positions], [p[1] for p in optimal_positions], 
         'g-', linewidth=2, label='Strategic Flapping')

# Draw birds at intervals along trajectories
for i in range(0, len(no_flap_positions), 8):
    plt.scatter(no_flap_positions[i][0], no_flap_positions[i][1], 
                color='red', s=100, alpha=0.7, zorder=5)
for i in range(0, len(flap_positions), 8):
    plt.scatter(flap_positions[i][0], flap_positions[i][1], 
                color='blue', s=100, alpha=0.7, zorder=5)
for i in range(0, len(optimal_positions), 8):
    plt.scatter(optimal_positions[i][0], optimal_positions[i][1], 
                color='green', s=100, alpha=0.7, zorder=5)

# Add a pipe to navigate through
# Top pipe
plt.fill_between(
    [pipe_x, pipe_x + 52], 
    [0, 0], 
    [gap_center - 50, gap_center - 50], 
    color='green', alpha=0.7
)
# Bottom pipe
plt.fill_between(
    [pipe_x, pipe_x + 52], 
    [gap_center + 50, gap_center + 50], 
    [512, 512], 
    color='green', alpha=0.7
)

# Add force vectors to show gravity and flap
# For gravity
plt.arrow(50, 100, 0, 20, width=2, head_width=10, head_length=10, 
          fc='black', ec='black', zorder=6)
plt.text(60, 110, 'Gravity\n(0.5)', fontsize=10)

# For flap
plt.arrow(150, 200, 0, -30, width=2, head_width=10, head_length=10, 
          fc='black', ec='black', zorder=6)
plt.text(160, 190, 'Flap\n(-8.0)', fontsize=10)

plt.title('Bird Physics Simulation', fontsize=16)
plt.legend(loc='upper right')
plt.grid(alpha=0.2)
plt.axis('off')

plt.tight_layout()
plt.savefig('Figures/bird_physics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Bird physics visualization saved!")

# 2. PIPE GENERATION VISUALIZATION
plt.figure(figsize=(12, 6))

# Background
plt.fill_between([0, 800], [0, 0], [512, 512], color='skyblue', alpha=0.3)
plt.fill_between([0, 800], [450, 450], [512, 512], color='tan')

# Generate a series of pipes with random heights
np.random.seed(42)  # For reproducibility
n_pipes = 6
pipe_width = 52
pipe_distance = 150
pipe_x_positions = [100 + i * pipe_distance for i in range(n_pipes)]
gap_centers = [200 + np.random.randint(-100, 100) for _ in range(n_pipes)]
gap_size = 100

# Draw pipes
for i, (pipe_x, gap_center) in enumerate(zip(pipe_x_positions, gap_centers)):
    # Top pipe
    plt.fill_between(
        [pipe_x, pipe_x + pipe_width], 
        [0, 0], 
        [gap_center - gap_size/2, gap_center - gap_size/2], 
        color='green', alpha=0.7
    )
    
    # Bottom pipe
    plt.fill_between(
        [pipe_x, pipe_x + pipe_width], 
        [gap_center + gap_size/2, gap_center + gap_size/2], 
        [512, 512], 
        color='green', alpha=0.7
    )
    
    # Add labels for pipe number and gap center
    plt.text(pipe_x + pipe_width/2, 30, f"Pipe {i+1}", ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    plt.text(pipe_x + pipe_width/2, gap_center, f"{gap_center}", ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

# Add measurement indicators
# Show pipe width
plt.annotate('', xy=(pipe_x_positions[0], 400), xytext=(pipe_x_positions[0] + pipe_width, 400),
             arrowprops=dict(arrowstyle='<->', color='black'))
plt.text(pipe_x_positions[0] + pipe_width/2, 410, f"Pipe Width\n({pipe_width} px)", ha='center')

# Show pipe distance
plt.annotate('', xy=(pipe_x_positions[0] + pipe_width, 350), 
             xytext=(pipe_x_positions[1], 350),
             arrowprops=dict(arrowstyle='<->', color='black'))
plt.text((pipe_x_positions[0] + pipe_width + pipe_x_positions[1])/2, 360, 
         f"Pipe Distance\n({pipe_distance - pipe_width} px)", ha='center')

# Show gap size
plt.annotate('', xy=(pipe_x_positions[2] + pipe_width + 20, gap_centers[2] - gap_size/2), 
             xytext=(pipe_x_positions[2] + pipe_width + 20, gap_centers[2] + gap_size/2),
             arrowprops=dict(arrowstyle='<->', color='black'))
plt.text(pipe_x_positions[2] + pipe_width + 40, gap_centers[2], 
         f"Gap Size\n({gap_size} px)", va='center')

plt.title('Pipe Generation and Parameters', fontsize=16)
plt.xlim(50, 700)
plt.ylim(512, 0)  # Invert y-axis
plt.axis('off')

plt.tight_layout()
plt.savefig('Figures/pipe_generation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Pipe generation visualization saved!")

# 3. ENVIRONMENT MODES COMPARISON
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Simple graphics mode (left)
ax = axes[0]
# Background
ax.fill_between([0, 288], [0, 0], [512, 512], color='skyblue', alpha=0.3)
ax.fill_between([0, 288], [450, 450], [512, 512], color='tan')

# Draw a pipe
pipe_x = 150
gap_center = 250
gap_size = 100
pipe_width = 52

# Top pipe
ax.fill_between(
    [pipe_x, pipe_x + pipe_width], 
    [0, 0], 
    [gap_center - gap_size/2, gap_center - gap_size/2], 
    color='green', alpha=0.7
)

# Bottom pipe
ax.fill_between(
    [pipe_x, pipe_x + pipe_width], 
    [gap_center + gap_size/2, gap_center + gap_size/2], 
    [512, 512], 
    color='green', alpha=0.7
)

# Bird
ax.scatter([100], [250], s=300, color='yellow', edgecolor='black', zorder=5)

ax.set_title('Simple Graphics Mode', fontsize=14)
ax.set_xlim(0, 288)
ax.set_ylim(512, 0)  # Invert y-axis
ax.axis('off')

# Graphic assets mode (right)
ax = axes[1]
# This is a mock-up of what the game would look like with assets
# In a real implementation, you'd load and display actual game assets

# Background (simulate a sky background with clouds)
ax.fill_between([0, 288], [0, 0], [512, 512], color='skyblue', alpha=0.5)
# Draw clouds
cloud_positions = [(50, 100), (150, 80), (220, 120)]
for cx, cy in cloud_positions:
    ellipse = plt.matplotlib.patches.Ellipse((cx, cy), 60, 30, color='white', alpha=0.8)
    ax.add_patch(ellipse)
    ellipse = plt.matplotlib.patches.Ellipse((cx+15, cy+10), 50, 25, color='white', alpha=0.9)
    ax.add_patch(ellipse)

# Draw textured ground
ax.fill_between([0, 288], [450, 450], [512, 512], color='#8B4513', alpha=0.7)
# Add texture lines to ground
for i in range(0, 288, 20):
    ax.plot([i, i+10], [460, 460], color='#654321', linewidth=2)

# Draw pipes with texture
pipe_x = 150
gap_center = 250
gap_size = 100
pipe_width = 52

# Top pipe
rect = plt.Rectangle((pipe_x, 0), pipe_width, gap_center - gap_size/2, 
                    facecolor='green', alpha=0.8, zorder=3)
ax.add_patch(rect)
# Pipe cap
rect = plt.Rectangle((pipe_x-5, gap_center - gap_size/2 - 10), pipe_width+10, 10, 
                    facecolor='forestgreen', alpha=0.9, zorder=4)
ax.add_patch(rect)

# Bottom pipe
rect = plt.Rectangle((pipe_x, gap_center + gap_size/2), pipe_width, 512 - (gap_center + gap_size/2), 
                    facecolor='green', alpha=0.8, zorder=3)
ax.add_patch(rect)
# Pipe cap
rect = plt.Rectangle((pipe_x-5, gap_center + gap_size/2), pipe_width+10, 10, 
                    facecolor='forestgreen', alpha=0.9, zorder=4)
ax.add_patch(rect)

# Bird with better styling (simulate a sprite)
bird_color = 'gold'
bird_x, bird_y = 100, 250

# Draw a more bird-like shape
bird_body = plt.matplotlib.patches.Ellipse((bird_x, bird_y), 34, 24, 
                                          color=bird_color, zorder=5)
ax.add_patch(bird_body)
# Add eye
eye = plt.matplotlib.patches.Circle((bird_x+10, bird_y-5), 3, 
                                   color='white', zorder=6)
ax.add_patch(eye)
pupil = plt.matplotlib.patches.Circle((bird_x+11, bird_y-5), 1.5, 
                                     color='black', zorder=7)
ax.add_patch(pupil)
# Add beak
beak_vertices = np.array([[bird_x+17, bird_y], [bird_x+25, bird_y-3], [bird_x+17, bird_y+3]])
beak = plt.Polygon(beak_vertices, closed=True, color='orange', zorder=6)
ax.add_patch(beak)
# Add wing
wing_vertices = np.array([[bird_x-5, bird_y+2], [bird_x+5, bird_y+8], [bird_x+7, bird_y+2]])
wing = plt.Polygon(wing_vertices, closed=True, color='goldenrod', zorder=6)
ax.add_patch(wing)

# Add score with nicer styling
ax.text(10, 30, "Score: 2", fontsize=16, fontweight='bold',
       bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='gray', alpha=0.8))

ax.set_title('Asset-Based Graphics Mode', fontsize=14)
ax.set_xlim(0, 288)
ax.set_ylim(512, 0)  # Invert y-axis
ax.axis('off')

plt.suptitle('Comparison of Environment Rendering Modes', fontsize=16)
plt.tight_layout()
plt.savefig('Figures/rendering_modes.png', dpi=300, bbox_inches='tight')
plt.close()
print("Environment modes comparison saved!")

print("\nAdditional environment visualizations completed!")