import numpy as np
import pandas as pd

from neural_network import *
from robo import *
from ga import *

df = pd.read_csv('data/reliance.csv')
df = df[:10000]

def main(ann):
    robos = {}

    ge = {}
    nets = {}
    for robo_id in range(0, len(ann)):
        nets[robo_id] = net
        robos[robo_id] = Robo(ann=ann[robo_id], df=df, money=100000)
        ge[robo_id] = robos[robo_id].fitness()
    return ge 

# np.random.seed(1999)
x = np.random.rand(13, 1)
y = np.random.rand(3, 1)
bestPop = GA(x, y, n_h=[20, 12], generations=10000, popSize=100, eliteSize=10, main=main, mutationRate=0.5)
# with open('weights/weights0.pickle', 'rb') as f:
#     x = pickle.load(f)
#     tmp = []
#     tmp.append(x)
#     print(main(tmp))
# with open('save.json', 'w') as fp:
#     json.dump(save.generationBest, fp)
# print(len(save.generationBest))