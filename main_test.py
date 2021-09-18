import numpy as np
import pandas as pd

from neural_network import NeuralNetwork
from robo import Robo_Test
from ga import GA

df = pd.read_csv('data/reliance.csv')
# 375 ind is start of 9:15 and 10088 time is 15:29
df = df[375:10088]
df.index = np.arange(0, df.shape[0])
df["Time"] = df.Date
df.Date = df.Date.apply(lambda x: (int(x[11:13])-9)*60 + int(x[14:16]) - 14)


def main_test(ann):
    robos = {}

    ge = {}
    nets = {}
    for robo_id in range(0, len(ann)):
        nets[robo_id] = net
        robos[robo_id] = Robo_Test(ann=ann[robo_id], df=df, money=100000)
        ge[robo_id] = robos[robo_id].fitness()
    return ge 


with open('weights.pickle', 'rb') as f:
    x = pickle.load(f)
    tmp = []
    tmp.append(x)
    print(main(tmp))