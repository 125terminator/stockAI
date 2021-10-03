import os
import time
import pickle

import neat
import numpy as np
import pandas as pd

from robo import Robo
from input import Inputs

df = pd.read_csv('./data/reliance.csv')
# 375 ind is start of 9:15 and 10088 time is 15:29
n = 375*41
df = df[:n]
df.index = np.arange(0, df.shape[0])
df["Time"] = df.Date
df.Date = df.Date.apply(lambda x: (int(x[11:13])-9)*60 + int(x[14:16]) - 14)


GEN_NUM = 0
BEST_FITNESS = 0
if not os.path.exists('log'):
    os.mkdir('log')
ann_inputs = Inputs(df)


def eval_genomes(genomes, config):
    global GEN_NUM, BEST_FITNESS
    GEN_NUM += 1
    logger = open('log/{}'.format(GEN_NUM), 'w')
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        robo = Robo(ann=net, df=df, money=100000, inputs=ann_inputs.inputs, logger=logger)
        genome.fitness = robo.fitness()
        if genome.fitness > BEST_FITNESS:
            pickle.dump(net,open("best.pickle", "wb"))
            BEST_FITNESS = genome.fitness
    logger.close()



def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward.txt')
    run(config_path)