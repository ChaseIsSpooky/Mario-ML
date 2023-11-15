from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from ddq_keras import DDQNAgent1
from utils import plotLearning
import numpy as np
import gym
import copy
from skimage.color import rgb2gray
from skimage.transform import resize
    
env = gym.make('SuperMarioBros-v0')
state = env.close()
 # pre-processing
 # pre-processing
 # Convert state to grayscale
state_gray = rgb2gray(state)
 # Downsample state to 60x64
state_downsampled = resize(state_gray, (60, 64))
# Flatten state to 1D array
state_flattened = state_downsampled.flatten()

q_table = np.zeros((env.observation_space.n, env.action_space.n))
q_target = np.zeros((env.observation_space.n, env.action_space.n))

class AgentDoubleQ():
   def __init__(self, alpha, gamma, copy_steps):
       self.alpha = alpha
       self.gamma = gamma
       self.copy_steps = copy_steps

   def q_update(self, state, action_id, reward, next_state, terminal):
       if terminal:
           target = reward
       else:
           target = reward + self.gamma*max(self.q_target[next_state])
       
       td_error = target - self.q_table[state, action_id]
       self.q_table[state, action_id] = self.q_table[state, action_id] + self.alpha*td_error
   
   def copy(self):
       self.q_target = copy.deepcopy(self.q_table)


def train_agent_doubleq(agent, env, num_episodes):
   for e in range(num_episodes):
       state = env.reset()
       while True:
           action_id = np.argmax(agent.q_table[state])
           next_state, reward, terminal = env.step(action_id)
           
           agent.q_update(state, action_id, reward, next_state, terminal)
           state = next_state
           
           if terminal:
               break
           
           if e % agent.copy_steps == 0:
               agent.copy()

def test_agent(agent, env):
   state = env.reset()
   while True:
       action_id = np.argmax(agent.q_table[state])
       next_state, reward, terminal = env.step(action_id)
       state = next_state
       if terminal:
           break

   

"""     done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()

    env.close() """

