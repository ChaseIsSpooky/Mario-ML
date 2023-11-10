from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from utils import plotLearning
import numpy as np
import gym
from ddq_keras import DDQNAgent
    

"""     done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()

    env.close() """

# Initialize the Mario environment
env = gym.make('SuperMarioBros-1-1-v0')
state_size = env.observation_space.shape[0]
action_size = len(COMPLEX_MOVEMENT)  

# Initialize DDQN agent
agent = DDQNAgent(state_size, action_size)

# Training parameters
batch_size = 32
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    print("state = ", state, "state_size = ", state_size)
    state_size = env.observation_space.shape
    state = np.reshape(state, [1, state_size])

    total_reward = 0
    for time in range(500): 
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward

        if done:
            print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(episode + 1, num_episodes, total_reward, agent.epsilon))
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)