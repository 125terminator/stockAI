import numpy as np
import random
from neural_network import NeuralNetwork
import time, datetime
from numpy import loadtxt
import operator
import copy
from random import randrange
from math import floor
from numpy import loadtxt, savetxt
import pickle

population = []
generationCount = 0
popRanked = {}
def GA(X, Y, n_h, main, generations=10, popSize=100, eliteSize=10, mutationRate=0.05):

  def initial_population(popSize):
    population=[]

    for i in range(popSize):
        population.append(NeuralNetwork(X, Y, n_h))
    return population

  def mutation(child, mutationRate):
    global generationCount
    scale = 1
    lscale = 0.0
    # Add decay in scale
    # scale = scale/pow(10, generationCount//10)
    for _, params in child.params.items():
      if random.random() <= mutationRate:
        params += np.random.normal(loc=lscale, scale=scale, size=params.shape)

  def rankPopulation():
    global population, popRanked
    popRanked = main(population)
    # fitnessSum=0
    # for i in range(len(population)):
    #   # print(population[i], population[i].food)
    #   fit = population[i].compute_cost(X, Y)
    #   # fitnessSum+=fit
    #   popRanked[i] = fit
    popRanked = sorted(popRanked.items(), key = operator.itemgetter(1), reverse = True)
    # return fitnessSum, rankedPopulation
  
  def random_pick():
    global popRanked
    return randrange(len(popRanked))
    parentSelectPercentage = 0.5
    l = 0
    r = floor(parentSelectPercentage * len(popRanked))
    return randrange(l, r+1)

  def next_generation(eliteSize, mutationRate):
    global population
    global popRanked
    # popRanked = rankPopulation()
    # fitnessSum = popRanked[0]
    newPopulation = []
    for i in range(eliteSize):
      newPopulation.append(population[popRanked[i][0]])
    for i in range(len(population)-eliteSize):
      tmpPop = copy.deepcopy(population[random_pick()])
      mutation(tmpPop, mutationRate)
      newPopulation.append(tmpPop)
    return newPopulation

  def genetic_algorithm(popSize, eliteSize, mutationRate, generations):
    global population, generationCount, popRanked
    generationCount = 0
    population = initial_population(popSize)
    # popRanked = rankPopulation()
    # print("Initial fitness: " + str(popRanked[0][1]))
    best_fitness = -1e9
    best_pop = []

    for i in range(generations):
      generationCount += 1
      rankPopulation()
      fitness = popRanked[0][1]
      if best_fitness < fitness:
        best_fitness = fitness
        best_pop = copy.deepcopy(population[popRanked[0][0]])
      print("Generation : {}\t Fitness: {}".format(str(i+1), str(fitness)))

      population = next_generation(eliteSize, mutationRate)
      if (i+1)%10==0:
        with open('weights.pickle', 'wb') as handle:
          pickle.dump(best_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return best_pop
  return genetic_algorithm(popSize, eliteSize, mutationRate, generations)
