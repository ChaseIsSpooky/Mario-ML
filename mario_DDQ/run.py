from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from ddq_keras import DDQNAgent1
from utils import plotLearning
import numpy as np
    
if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    ddqn_agent = DDQNAgent1(alpha=.0005, gamma= 0.99, n_actions=12, epsilon=1.0,
                           batch_size=64,input_dims=8)
    
    n_games = 500
    #ddqn_agent.load_model()
    ddqn_scores = []
    eps_history = []

    
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = ddqn_agent.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            ddqn_agent.learn()
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        if i%10 == 0 and i > 0:
            ddqn_agent.save_model()

    filename = 'mario-ddqn.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)

   

"""     done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()

    env.close() """

