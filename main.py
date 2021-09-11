
GEN_NUM = 0

def main(neuralNetwork):
    ##########
    global GEN_NUM
    robos = {}
    
    best_fitness = -1e6
    best_robo = Robo

    ge = {}
    nets = {}
    robo_id = 0
    for net in neuralNetwork:
        nets[robo_id] = net
        robos[robo_id] = Robo()
        ge[robo_id] = 0
        robo_id += 1
    ##########
    QUIT = False
    while QUIT == False:
        rem_robos = set()
        for robo_id, robo in robos.items():
            brainInput = robo.getInputs()
            brainInput = brainInput.reshape(brainInput.shape[0], 1)
            brainOutput = nets[robo_id].forward_propagation(brainInput)
            performance, isIn = robo.move(brainOutput)
            ge[robo_id] += performance
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
x = np.random.rand(34, 1)
y = np.random.rand(9, 1)
bestPop = GA(x, y, n_h=[20, 12], generations=10000, popSize=100, eliteSize=10, main=main, mutationRate=0.5)
# with open('weights/weights0.pickle', 'rb') as f:
#     x = pickle.load(f)
#     tmp = []
#     tmp.append(x)
#     print(main(tmp))
# with open('save.json', 'w') as fp:
#     json.dump(save.generationBest, fp)
# print(len(save.generationBest))