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

def evaluate_network(genome, config, env):
    # Create a neural network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    state = env.reset()  # Extract the observation from the reset result
    total_reward = 0
    done = False

    while not done:
        # Process the state through the neural network to get actions
        action = np.argmax(net.activate(state))
        print(action)
        state, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward

    return total_reward


def run(config):

    # Create the Gym environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run the NEAT evolution loop
    generations = 100
    for generation in range(generations):
        print(f"Generation {generation + 1}")

        # Evaluate each agent in the population
        fitness_scores = []
        for genome_id, genome in p.population.items():
            fitness = evaluate_network(genome, config, env)
            fitness_scores.append((genome_id, fitness))

        for genome_id, fitness in fitness_scores:
            p.population[genome_id].fitness = fitness

        # Run NEAT's evolution step
        p.evolve()

        best_genome = p.best_genome()
        best_fitness = best_genome.fitness
        print(f"Best Fitness: {best_fitness}")

    return p  # Return the population object

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "Neat_config.txt")
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)
    population = run(config)  # Store the population object returned by the run function

    # Get the best-performing network and use it to play the game
    best_genome = population.best_genome()
    best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)
    while True:
        evaluate_network(best_genome, config)
