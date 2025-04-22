import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create example data for hyperparameter sensitivity
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
network_sizes = [16, 32, 64, 128, 256]

# Example performance matrix (replace with your actual data if available)
performance = np.array([
    [5.2, 7.8, 10.3, 11.2, 10.8],   # LR = 0.0001
    [8.5, 12.3, 15.7, 14.9, 13.2],  # LR = 0.0005
    [11.2, 13.8, 14.1, 12.5, 9.8],  # LR = 0.001
    [9.7, 11.2, 8.9, 7.3, 5.1],     # LR = 0.005
    [6.3, 7.5, 5.2, 3.8, 2.6]       # LR = 0.01
])

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(performance, annot=True, cmap='viridis', 
            xticklabels=network_sizes, yticklabels=learning_rates)
plt.xlabel('Network Size (Neurons per Hidden Layer)', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Performance Sensitivity to Hyperparameters', fontsize=14)
plt.tight_layout()
plt.savefig('figures/hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()

print("Hyperparameter sensitivity figure saved!")