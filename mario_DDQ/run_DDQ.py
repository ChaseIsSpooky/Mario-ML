import tensorflow as tf
from keras import layers
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

model = tf.keras.Sequential([
    layers.Input(shape=(240, 256, 3)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(env.action_space.n, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
class AgentDoubleQ():
    def __init__(self, alpha, gamma, copy_steps):
        self.alpha = alpha
        self.gamma = gamma
        self.copy_steps = copy_steps
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(240, 256, 3)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(env.action_space.n, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def q_update(self, state, action_id, reward, next_state, terminal):
        if terminal:
            target = reward
        else:
            # Use the target network for Q-value prediction
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state))

        # Update the Q-value of the chosen action
        q_values = self.model.predict(state)
        q_values[0][action_id] = target

        # Train the model on the updated Q-values
        self.model.fit(state, q_values, verbose=0)

    def copy(self):
        # Update the target network weights with the current model weights
        self.target_model.set_weights(self.model.get_weights())

def train_agent_doubleq(agent, env, num_episodes, epsilon):
    for e in range(num_episodes):
        state = env.reset()
        state_gray = rgb2gray(state)
        state_downsampled = resize(state_gray, (60, 64))
        state_flattened = state_downsampled.flatten()
        # x = True
        # y = 200
        tot_rew1 = 0
        while True:
            print("im in train")
            # Epsilon-greedy exploration
            if np.random.rand() < epsilon:
                action_id = env.action_space.sample()
            else:
                q_values = agent.model.predict(state.reshape(1, 240, 256, 3))
                action_id = np.argmax(q_values)

            next_state, reward, terminal, info = env.step(action_id)
            print(terminal)
            print(info)
            print("total reward: ")
            tot_rew1 += reward
            print(tot_rew1)
            next_state_gray = rgb2gray(next_state)
            next_state_downsampled = resize(next_state_gray, (60, 64))
            next_state_flattened = next_state_downsampled.flatten()

            agent.q_update(state.reshape(1, 240, 256, 3), action_id, reward, next_state.reshape(1, 240, 256, 3), terminal)
            state = next_state
            env.render()
            # y = y-1
            # if(y==0):
            #     x=False
            if terminal:
                tot_rew1 = 0
                break

            if e % agent.copy_steps == 0:
                agent.copy()

def test_agent(agent, env):
    state = env.reset()
    tot_rew2 = 0
    while True:
        print("im in test")
        q_values = agent.model.predict(state.reshape(1, 240, 256, 3))
        action_id = np.argmax(q_values)
        next_state, reward, terminal, info = env.step(action_id)
        print(terminal)
        print(info)
        print("total reward: ")
        tot_rew2 += reward
        print(tot_rew2)
        state = next_state
        env.render()
        tot_rew2 = 0
        if terminal:
            
            break

def __main__():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    alpha = 0.001  # Learning rate .001
    gamma = 0.99   # Discount factor .99
    copy_steps = 100  # Frequency of updating the target network 100
    epsilon = .1  # Exploration rate .1

    agent = AgentDoubleQ(alpha, gamma, copy_steps)

    num_train_episodes = 100 # 100
    num_test_episodes = 10 # 10

    # Training
    train_agent_doubleq(agent, env, num_train_episodes, epsilon)

    # Testing
    for _ in range(num_test_episodes):
        test_agent(agent, env)

if __name__ == "__main__":
    __main__()