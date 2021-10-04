import numpy as np
from typing import List, Union, Optional, Tuple


def simulated_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This crossover is specific to floating-point representation.
    Simulate behavior of one-point crossover for binary representations.

    For large values of eta there is a higher probability that offspring will be created near the parents.
    For small values of eta, offspring will be more distant from parents

    Equation 9.9, 9.10, 9.11
    @TODO: Link equations
    """    
    # Calculate Gamma (Eq. 9.11)
    rand = np.random.random(parent1.shape)
    gamma = np.empty(parent1.shape)
    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))  # First case of equation 9.11
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))  # Second case

    # Calculate Child 1 chromosome (Eq. 9.9)
    chromosome1 = 0.5 * ((1 + gamma)*parent1 + (1 - gamma)*parent2)
    # Calculate Child 2 chromosome (Eq. 9.10)
    chromosome2 = 0.5 * ((1 - gamma)*parent1 + (1 + gamma)*parent2)

    return chromosome1, chromosome2

def gaussian_mutation(chromosome: np.ndarray, prob_mutation: float,
                    mu: List[float] = None, sigma: List[float] = None,
                    scale: Optional[float] = None) -> None:
  """
  Perform a gaussian mutation for each gene in an individual with probability, prob_mutation.

  If mu and sigma are defined then the gaussian distribution will be drawn from that,
  otherwise it will be drawn from N(0, 1) for the shape of the individual.
  """
  # Determine which genes will be mutated
  mutation_array = np.random.random(chromosome.shape) < prob_mutation
  # If mu and sigma are defined, create gaussian distribution around each one
  if mu and sigma:
      gaussian_mutation = np.random.normal(mu, sigma)
  # Otherwise center around N(0,1)
  else:
      gaussian_mutation = np.random.normal(size=chromosome.shape)
  
  if scale:
      gaussian_mutation[mutation_array] *= scale

  # Update
  chromosome[mutation_array] += gaussian_mutation[mutation_array]

def random_uniform_mutation(chromosome: np.ndarray, prob_mutation: float,
                            low: Union[List[float], float],
                            high: Union[List[float], float]
                            ) -> None:
    """
    Randomly mutate each gene in an individual with probability, prob_mutation.
    If a gene is selected for mutation it will be assigned a value with uniform probability
    between [low, high).

    @Note [low, high) is defined for each gene to help get the full range of possible values
    @TODO: Eq 11.4
    """
    assert type(low) == type(high), 'low and high must have the same type'
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    if isinstance(low, list):
        uniform_mutation = np.random.uniform(low, high)
    else:
        uniform_mutation = np.random.uniform(low, high, size=chromosome.shape)
    chromosome[mutation_array] = uniform_mutation[mutation_array]

def single_point_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, major='r') -> Tuple[np.ndarray, np.ndarray]:
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    rows, cols = parent2.shape
    row = np.random.randint(0, rows)
    col = np.random.randint(0, cols)

    if major.lower() == 'r':
        offspring1[:row, :] = parent2[:row, :]
        offspring2[:row, :] = parent1[:row, :]

        offspring1[row, :col+1] = parent2[row, :col+1]
        offspring2[row, :col+1] = parent1[row, :col+1]
    elif major.lower() == 'c':
        offspring1[:, :col] = parent2[:, :col]
        offspring2[:, :col] = parent1[:, :col]

        offspring1[:row+1, col] = parent2[:row+1, col]
        offspring2[:row+1, col] = parent1[:row+1, col]

    return offspring1, offspring2

def crossover(parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None
        _SBX_eta = 100
        _SPBX_type = 'r'

        # SBX
        if randrange(2) == 0:
            child1_weights, child2_weights = simulated_binary_crossover(parent1_weights, parent2_weights, _SBX_eta)
            child1_bias, child2_bias =  simulated_binary_crossover(parent1_bias, parent2_bias, _SBX_eta)

        # Single point binary crossover (SPBX)
        else:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights, major=_SPBX_type)
            child1_bias, child2_bias =  single_point_binary_crossover(parent1_bias, parent2_bias, major=_SPBX_type)

        return child1_weights, child2_weights, child1_bias, child2_bias

def mutation(child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray, mutation_rate):
    scale = .2

    rand_mutation = randrange(2)
    # Gaussian
    if rand_mutation == 0:
        # Mutate weights
        gaussian_mutation(child1_weights, mutation_rate, scale=scale)
        gaussian_mutation(child2_weights, mutation_rate, scale=scale)

        # Mutate bias
        gaussian_mutation(child1_bias, mutation_rate, scale=scale)
        gaussian_mutation(child2_bias, mutation_rate, scale=scale)
    
    # Uniform random
    else:
        # Mutate weights
        random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
        random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

        # Mutate bias
        random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
        random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

def mating(p1, p2, mutation_rate):
    L = len(p1.layer_nodes)
    c1_params = {}
    c2_params = {}
    for l in range(1, L):
      p1_W_l = p1.params['W' + str(l)]
      p2_W_l = p2.params['W' + str(l)]  
      p1_b_l = p1.params['b' + str(l)]
      p2_b_l = p2.params['b' + str(l)]

      # Crossover
      # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
      c1_W_l, c2_W_l, c1_b_l, c2_b_l = crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

      # Mutation
      # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
      mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l, mutation_rate)

      # Assign children from crossover/mutation
      c1_params['W' + str(l)] = c1_W_l
      c2_params['W' + str(l)] = c2_W_l
      c1_params['b' + str(l)] = c1_b_l
      c2_params['b' + str(l)] = c2_b_l

      # Clip to [-1, 1]
      np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
      np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
      np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
      np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

    p1.params = c1_params
    p2.params = c2_params