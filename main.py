from util import ANN_Inputs
import numpy as np
import pandas as pd

from neural_network import NeuralNetwork
from robo import Robo
from ga import GA
from input import Inputs

df = pd.read_csv('data/reliance.csv')
# 375 ind is start of 9:15 and 10088 time is 15:29
df = df[:10088]
df.index = np.arange(0, df.shape[0])
df["Time"] = df.Date
df.Date = df.Date.apply(lambda x: (int(x[11:13])-9)*60 + int(x[14:16]) - 14)

ann_inputs = Inputs(df)

def main(ann):
    robos = {}

    ge = {}
    for robo_id in range(0, len(ann)):
        robos[robo_id] = Robo(ann=ann[robo_id], df=df, money=100000, inputs=ann_inputs)
        ge[robo_id] = robos[robo_id].fitness()
    return ge 

# np.random.seed(1999)
x = np.random.rand(14, 1)
y = np.random.rand(3, 1)
bestPop = GA(x, y, n_h=[20, 12], generations=10000, popSize=100, eliteSize=10, main=main, mutationRate=0.5)