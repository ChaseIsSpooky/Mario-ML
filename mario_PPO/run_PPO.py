import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT
from skimage.color import rgb2gray
from skimage.transform import resize
import gym_super_mario_bros
import gym
import time
import warnings
import os
import logging

# Filter out Gym-related warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")

# Define a simple neural network for the policy
class Policy:
    def __init__(self, input_size, output_size):
        self.weights_fc = np.random.randn(input_size, 128) / np.sqrt(input_size)
        self.weights_action_head = np.random.randn(128, output_size) / np.sqrt(128)

    def forward(self, x):
        x = np.maximum(0, np.dot(x, self.weights_fc))
        action_probs = np.exp(np.dot(x, self.weights_action_head))
        return action_probs / np.sum(action_probs)

# Function to compute discounted rewards
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    running_add = 0
    for r in reversed(rewards):
        running_add = running_add * gamma + r
        discounted_rewards.append(running_add)
    return list(reversed(discounted_rewards))

# PPO training function
def train_ppo(env, policy, epochs=1000, batch_size=64, gamma=0.99, epsilon=0.2, clip_value=0.1):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    input_size = 60 * 64  
    output_size = len(COMPLEX_MOVEMENT)  

    print("starting training")
    for epoch in range(epochs):
        total_reward = 0
        state = env.reset()
        states = []
        actions = []
        rewards = []


        state = env.reset()
        done = False
        while not done:
            state_gray = rgb2gray(state)
            state_downsampled = resize(state_gray, (60, 64))
            state_flattened = state_downsampled.flatten()

            # Get action probabilities from the policy
            action_probs = policy.forward(state_flattened)
            action = np.random.choice(range(output_size), p=action_probs)
            #print(action)

            # Take the action in the environment
            state, reward, done, info = env.step(action)
            #print(reward)

            # Save state, action, and reward for training
            states.append(state_flattened)
            actions.append(action)
            rewards.append(reward)
            
          

            if info['time'] <= 300:
                logging.info(f"Episode: {epoch}, Score: {info['score']}   x-position:  {info['x_pos']}")
                done = True
            env.render()


        total_reward = sum(rewards)
        print(f"Epoch {epoch + 1}, Total Reward: {total_reward}")
        

        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        print("computed discount_rewards")

        # Normalize discounted rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        print("normalized discount_rewards")

        # Convert states and actions to NumPy arrays
        states_array = np.array(states)
        actions_array = np.array(actions)
        discounted_rewards_array = np.array(discounted_rewards)
        print("converted states and actions to numpy arrays")

        # Compute advantages
        advantages = discounted_rewards_array
        print("computed advantages")

        # PPO optimization step
        print("starting ppo optimization")
        for _ in range(5):  # Number of optimization steps
            # Get action probabilities from the policy
            action_probs = policy.forward(states_array)

            # Compute ratio and surrogate loss
            ratios = np.exp(np.log(action_probs[range(len(action_probs)), actions_array]) - np.log(action_probs.max(axis=1)))
            surrogate_loss = -np.minimum(ratios * advantages, np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantages)

    # Update policy
    grad_fc = np.dot(states_array.T, np.diag(surrogate_loss))
    grad_action_head = np.dot(policy.weights_fc.T, np.diag(surrogate_loss))

    policy.weights_fc -= 1e-3 * grad_fc
    policy.weights_action_head -= 1e-3 * grad_action_head

if __name__ == "__main__":
    # Example usage
    input_size = 60*64
    output_size = len(COMPLEX_MOVEMENT)
    env = gym.make('SuperMarioBros-v0')
    policy = Policy(input_size, output_size)
    local_dir = os.path.dirname(__file__)
    log_file = os.path.join(local_dir, "PPO_graph_data.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    train_ppo(env, policy)

    