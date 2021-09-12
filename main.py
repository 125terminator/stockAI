
GEN_NUM = 0

def main(ann):
    ##########
    global GEN_NUM
    robos = {}

    ge = {}
    nets = {}
    for robo_id in range(0, len(ann)):
        nets[robo_id] = net
        robos[robo_id] = Robo(ann[robo_id], 100000)
        ge[robo_id] = robos[robo_id].fitness()
    # fp_fitness.write('{}\n'.format(best_fitness))
    # fp_food.write('{}\n'.format(best_robo.num_food))
    # if GEN_NUM%10 == 0:
    #     if not os.path.exists('save'):
    #       os.makedirs('save')
    #     saveDict = {'robo' : best_robo.robo_locations, 'food' : best_robo.food_locations}
    #     # save.generationBest[GEN_NUM] = saveDict
    #     with open('save/save{}.json'.format(GEN_NUM), 'w') as fp:
    #         json.dump(saveDict, fp)
    GEN_NUM += 1
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