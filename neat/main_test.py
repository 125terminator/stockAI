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

if not os.path.exists('log'):
    os.mkdir('log')
ann_inputs = Inputs(df)


def eval_genomes(genomes):
    logger = open('log/{}'.format('neat_test'), 'w')
    for genome_id, genome in genomes:
        net = genome
        robo = Robo(ann=net, df=df, money=100000, inputs=ann_inputs.inputs, logger=logger)
        genome.fitness = robo.fitness()
    logger.close()


with open('best.pickle', 'rb') as f:
    x = pickle.load(f)
    tmp = {}
    tmp[0] = x
    print(eval_genomes(tmp))