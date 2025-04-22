import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Increase from 24 to 64 neurons
        self.fc2 = nn.Linear(64, 64)          # Increase from 24 to 64 neurons
        self.fc3 = nn.Linear(64, 32)          # Add an extra layer
        self.fc4 = nn.Linear(32, action_size)
        self.dropout = nn.Dropout(0.2)        # Add dropout for regularization
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size  # Size of state vector (5 features)
        self.action_size = action_size  # Number of actions (2: do nothing, flap)
        self.device = device  # CPU or GPU
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005  # Learning rate
        
        # Memory for experience replay
        self.memory = deque(maxlen=50000)
        
        # Build the neural network model
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action according to the model
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()
        return torch.argmax(act_values, dim=1).item()
    
    def replay(self, batch_size):
        """Train the model with random samples from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            target_q_values = target_q_values.unsqueeze(1)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon for less exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights from file"""
        if os.path.isfile(name):
            self.model.load_state_dict(torch.load(name))
            self.target_model.load_state_dict(torch.load(name))
    
    def save(self, name):
        """Save model weights to file"""
        torch.save(self.model.state_dict(), name)