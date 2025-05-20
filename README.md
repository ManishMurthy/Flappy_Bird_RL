Flappy Bird Using Deep Reinforcement Learning

Overview

This project implements a Deep Q-Network (DQN) agent that learns to play Flappy Bird autonomously through reinforcement learning. The agent learns optimal behavior through trial and error, determining when to flap its wings and when to glide to maximize survival time.
Features

Custom Pygame-based Flappy Bird environment
Deep Q-Network (DQN) implementation with PyTorch
Experience replay mechanism for stable learning
Epsilon-greedy exploration strategy with decay
Compact state representation with 5 key features
Visualization tools for agent performance and decision boundaries
Saved model checkpoints for continuing training
Comprehensive performance analysis

Installation
# Clone the repository
git clone https://github.com/yourusername/Flappy_Bird_RL.git
cd Flappy_Bird_RL

# Install dependencies
pip install -r requirements.txt
Requirements

Python 3.11+
PyTorch
Pygame
NumPy
Matplotlib

Usage
# Train the agent from scratch
python train.py

# Test a pre-trained agent
python test.py --model_path checkpoints/best_model.pth

# Visualize agent performance
python visualize.py --model_path checkpoints/best_model.pth
Project Structure
Flappy_Bird_RL/
├── flappy_env.py     # Flappy Bird game environment
├── train.py          # Training script
├── test.py           # Testing script
├── dqn_agent.py      # Agent implementation
└── Figures/          # Visualizations and figures

Results
Our DQN agent achieves an average score of 15.7 pipes after 5000 training episodes, significantly outperforming random play (0.01) and rule-based approaches (4.3). The agent demonstrates sophisticated understanding of game physics and employs strategic flapping patterns for optimal navigation.

Citation
If you use this code in your research, please cite our paper:
@article{murthy2025flappybird,
  title={Flappy Bird Using Deep Reinforcement Learning: A Deep Q-Network Approach},
  author={Murthy, Manish and Lokesh, Sanjana Mandya},
  journal={[Journal or Conference]},
  year={2025}
}

[![Flappy Bird RL CI](https://github.com/ManishMurthy/Flappy_Bird_RL/actions/workflows/python-app.yml/badge.svg)](https://github.com/ManishMurthy/Flappy_Bird_RL/actions/workflows/python-app.yml)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
