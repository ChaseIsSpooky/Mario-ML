from nes_py.wrappers import JoypadSpace
import neat
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import os
from skimage.color import rgb2gray
from skimage.transform import resize

def eval_genomes(genomes, config):
    best_genome_path = "best_genomes.txt"
    best_fit = -99999
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness, score, time, flag = run_mario(net)
        if genome.fitness > best_fit:
            best_fit = genome.fitness
            
    # Save the best genome of each generation to the file
    with open(best_genome_path, "a") as f:
        f.write(f"Generation: {p.generation}\n")
        f.write(f"Best Genome ID: {genome_id}\n") #need this to be something that will allow us to replay it visually after the algorithm
        f.write(f"Best Fitness Score: {best_fit}\n")
        f.write(f"Best in game score: {score}\n")
        if flag == True:
            f.write(f"Best in game time (only if flag is gotten): {time}\n")
        f.write("\n")
    

def run_mario(net):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    state = env.reset()
    done = False
    total_reward = 0
    print("next genome")

    while not done:
        # pre-processing
        # Convert state to grayscale
        state_gray = rgb2gray(state)
        # Downsample state to 60x64
        state_downsampled = resize(state_gray, (60, 64))
        # Flatten state to 1D array
        state_flattened = state_downsampled.flatten()
        output = net.activate(state_flattened)
        action = np.argmax(output)
        action = min(max(action, 0), 11)
        #print(action)
        state, reward, done, info = env.step(action)
        if info['time'] <= 0:
            done = True
        #env.render()
        total_reward += reward
    print(total_reward)

    return total_reward, info['score'], info['time'], info['flag_get']

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "Neat_Config.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Add a reporter to save the best genome of each generation
    best_genome_reporter = neat.Checkpointer(generation_interval=1, time_interval_seconds=None, filename_prefix="best_genome_")
    p.add_reporter(best_genome_reporter)
    
    winner = p.run(eval_genomes, 50)
