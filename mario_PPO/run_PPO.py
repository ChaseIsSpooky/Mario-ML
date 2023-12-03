import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from skimage.color import rgb2gray
from skimage.transform import resize
import gym
import time
import warnings

# Filter out Gym-related warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")

# Define a simple neural network for the policy
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.action_head = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

# Function to compute discounted rewards
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    running_add = 0
    for r in reversed(rewards):
        running_add = running_add * gamma + r
        discounted_rewards.append(running_add)
    return list(reversed(discounted_rewards))

# PPO training function
def train_ppo(env, policy, optimizer, epochs=1000, batch_size=64, gamma=0.99, epsilon=0.2, clip_value=0.1):
    input_size = 60 * 64  
    output_size = len(SIMPLE_MOVEMENT)  

    print("starting training")
    for epoch in range(epochs):
        
        state = env.reset()
        states = []
        actions = []
        rewards = []

        total_reward = sum(rewards)
        print(f"Epoch {epoch + 1}, Total Reward: {total_reward}")

        render_interval = 10
        if epoch % render_interval == 0:
        #     env.render()
            print(f"saving epoch: {epoch}")
            torch.save(policy.state_dict(), f"ppo_model_epoch_{epoch}.pth")

            

        state = env.reset()
        print("entering training loop")
        while True:
            state_gray = rgb2gray(state)
            state_downsampled = resize(state_gray, (60, 64))
            state_flattened = state_downsampled.flatten()

            # Convert state to PyTorch tensor
            state_tensor = torch.tensor(state_flattened, dtype=torch.float32)

            # Get action probabilities from the policy
            action_probs = policy(state_tensor)
            action_distribution = Categorical(action_probs)
            action = action_distribution.sample()

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action.item())

            # Save state, action, and reward for training
            states.append(state_flattened)
            actions.append(action)
            rewards.append(reward)

            if done:
                break

            state = next_state

        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        print("computed discount_rewards")

        # Normalize discounted rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        print("normalized discount_rewards")

        # Convert states and actions to PyTorch tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        print("converted states and actions to pytorch tensors")

        # Compute advantages
        advantages = discounted_rewards_tensor
        print("computed advantages")

        # PPO optimization step
        print("starting ppo optimization")
        for _ in range(5):  # Number of optimization steps
            # Compute action probabilities and log probabilities
            action_probs = policy(states_tensor)
            action_distribution = Categorical(action_probs)
            log_probs = action_distribution.log_prob(actions_tensor)

            # Compute ratio and surrogate loss
            ratio = torch.exp(log_probs - log_probs.detach())
            surrogate_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages)

            # Update policy
            optimizer.zero_grad()
            loss = torch.mean(surrogate_loss)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    # Example usage
    input_size = 60*64
    output_size = len(SIMPLE_MOVEMENT)
    env = gym.make('SuperMarioBros-v0')
    policy = Policy(input_size, output_size)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    train_ppo(env, policy, optimizer)
