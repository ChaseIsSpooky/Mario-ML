# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, COMPLEX_MOVEMENT)

# done = True
# for step in range(5000):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()

# env.close()

from nes_py.wrappers import JoypadSpace
import neat
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import os

# Create the Gym environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def evaluate_network(genomes, config):
    for genome_id, genome in genomes:
        # Create a neural network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        state = env.reset()  # get the observation from the reset result
        total_reward = 0
        done = False
        iterations = 0

        while not done and iterations < 2000:
            action = np.argmax(net.activate(state))
            print(action)
            state, reward, done, _ = env.step(action)
            #env.render()
            total_reward += reward
            iterations += 1
        genome.fitness = total_reward

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 100 generations.
    winner = p.run(evaluate_network, 5)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "Neat_config.txt")
    run(config_path)  # Store the population object returned by the run function

