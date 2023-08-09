import random
import math
from cacheout.lru import LRUCache
from scipy.interpolate import RBFInterpolator

class Chromosome:
    def __init__(self, genetic_code, fitness):
      self.genetic_code = genetic_code
      self.fitness = fitness

    def __str__(self):
      return str(self.genetic_code) + " " + str(self.fitness)

    def __repr__(self):
      return self.__str__()

    def __eq__(self, other):
      return self.genetic_code == other.genetic_code

    def __hash__(self):
      return hash(self.genetic_code)

class MDSGeneticAlgorithm:
    def __init__(self, graph, seed = None,
                 iterations = None,
                 population_size = None,
                 reproduction_size = None,
                 tournament_size = None,
                 mutation_rate = None,
                 elitism_size = None):
      self.n = len(graph)
      if self.n == 0 or self.n == 1:
        return

      self.m = 2 ** self.n

      self.graph = []
      self.incoming_neighbors = [[] for _ in range(self.n)]
      # graph transposing
      for i in range(self.n):
        set = 0
        for j in range(self.n):
          if graph[j] & (1 << self.n - i - 1):
            set |= (1 << self.n - j - 1)
            # incoming neighbors list will be in reverse
            self.incoming_neighbors[self.n - j - 1].append(i)
        self.graph.append(set)

      if seed != None:
        random.seed(seed)
      
      self.nodes_order_by_in_degree = list(range(self.n))
      self.nodes_order_by_in_degree.sort(key = lambda x: len(self.incoming_neighbors[x]), reverse = True)
      self.nodes_order_by_in_degree = list(map(lambda x: self.n - 1 - x, self.nodes_order_by_in_degree))

      self.cache = LRUCache(self.n)

      self.mutation_rate = 0.5

      graph_sizes = [[10], [20], [50], [75], [100], [250], [10_000]]
      parameters = [[24, 28, 8, 8, 3], [54, 65, 16, 15, 4],
              [81, 153, 47, 25, 7], [129, 219, 66, 30, 8],
              [163, 285, 86, 35, 11], [193, 455, 173, 65, 16],
              [3000, 800, 100, 50, 14]]

      approximate_parameters = RBFInterpolator(graph_sizes, parameters)([[self.n]])

      self.iterations = int(approximate_parameters[0][0])
      self.population_size = int(approximate_parameters[0][1])
      self.reproduction_size = int(approximate_parameters[0][2])
      self.tournament_size = int(approximate_parameters[0][3])
      self.elitism_size = int(approximate_parameters[0][4])

      if self.iterations > 10_000:
        self.iterations = 10_000

      if self.population_size > 4000:
        self.population_size = 4000

      if self.reproduction_size > 1000:
        self.reproduction_size = 1000

      if self.tournament_size > 400:
        self.tournament_size = 400

      if self.elitism_size > 100:
        self.elitism_size = 100

      if iterations != None:
        self.iterations = iterations

      if population_size != None:
        self.population_size = population_size

      if reproduction_size != None:
        self.reproduction_size = reproduction_size

      if tournament_size != None:
        self.tournament_size = tournament_size
      
      if mutation_rate != None:
        self.mutation_rate = mutation_rate

      if elitism_size != None:
        self.elitism_size = elitism_size

    def valid_solution(self, genetic_code, index):
      for i in self.incoming_neighbors[index]:
        if self.graph[i] & genetic_code == 0:
          return False
      return True
    
    def calculate_fitness(self, genetic_code):
      fitness = self.cache.get(genetic_code, -1)
      if fitness != -1:
        return fitness
      helper_number = genetic_code
      fitness = 0
      while helper_number:
        helper_number &= (helper_number-1)
        fitness += 1
      self.cache.set(genetic_code, fitness)
      return fitness

    def chromosome_fix(self, chromosome):
      for index in range(self.n):
        if chromosome.genetic_code & self.graph[index] == 0:
          chromosome.genetic_code |= (1 << self.n - 1 - index)
          chromosome.fitness += 1
    
    def better_chromosome_fix(self, chromosome):
      for index in self.nodes_order_by_in_degree:
        if chromosome.genetic_code & self.graph[index] == 0:
          chromosome.genetic_code |= (1 << self.n - 1 - index)
          chromosome.fitness += 1

    def random_order_fix(self, chromosome):
      order = list(range(self.n))
      random.shuffle(order)
      for index in order:
        if chromosome.genetic_code & self.graph[index] == 0:
          chromosome.genetic_code |= (1 << self.n - 1 - index)
          chromosome.fitness += 1
    
    def best_chromosome_fix(self, removed_nodes, crossover_point, mutation_point, child):
        length_of_potentially_nondominated_nodes_list = 0
        removed_nodes_indexes = []
        
        if crossover_point != None:
          while removed_nodes:
            new_removed_nodes = removed_nodes & (removed_nodes - 1)
            index = round(math.log2(removed_nodes - new_removed_nodes))
            removed_nodes_indexes.append(index)
            length_of_potentially_nondominated_nodes_list += len(self.incoming_neighbors[index])
            removed_nodes = new_removed_nodes
        
        if mutation_point != None:
          if child.genetic_code & (1 << mutation_point) == 0:
            child.fitness -= 1
            if mutation_point not in removed_nodes_indexes:
              removed_nodes_indexes.append(mutation_point)
          else:
            child.fitness += 1

        if length_of_potentially_nondominated_nodes_list < self.n:
          for index in removed_nodes_indexes:
            for i in self.incoming_neighbors[index]:
              if self.graph[i] & child.genetic_code == 0:
                child.genetic_code |= (1 << (self.n - 1 - i))
                child.fitness += 1
        else:
          self.better_chromosome_fix(child)

    def create_children(self, parent1, parent2):
        child1_genetic_code, child2_genetic_code, crossover_point = self.crossover(parent1.genetic_code, parent2.genetic_code)
        child1_genetic_code, mutation_point1 = self.mutation(child1_genetic_code)
        child2_genetic_code, mutation_point2 = self.mutation(child2_genetic_code)

        child1_fitness = None
        child2_fitness = None

        removed_nodes1 = 0
        removed_nodes2 = 0

        if crossover_point != None:
          if crossover_point > self.n / 2:
            # negated implication
            removed_nodes1 = parent1.genetic_code & ~child1_genetic_code
            removed_nodes2 = parent2.genetic_code & ~child2_genetic_code

            parent1_second_half_fitness = self.calculate_fitness(parent1.genetic_code & ((1 << self.n - crossover_point) - 1))
            parent1_first_half_fitness = parent1.fitness - parent1_second_half_fitness

            parent2_second_half_fitness = self.calculate_fitness(parent2.genetic_code & ((1 << self.n - crossover_point) - 1))
            parent2_first_half_fitness = parent2.fitness - parent2_second_half_fitness

          else:
            removed_nodes1 = parent2.genetic_code & ~child1_genetic_code
            removed_nodes2 = parent1.genetic_code & ~child2_genetic_code

            parent1_first_half_fitness = self.calculate_fitness((parent1.genetic_code >> self.n - crossover_point) << (self.n - crossover_point))
            parent1_second_half_fitness = parent1.fitness - parent1_first_half_fitness

            parent2_first_half_fitness = self.calculate_fitness((parent2.genetic_code >> self.n - crossover_point) << (self.n - crossover_point))
            parent2_second_half_fitness = parent2.fitness - parent2_first_half_fitness

          child1_fitness = parent1_first_half_fitness + parent2_second_half_fitness
          child2_fitness = parent2_first_half_fitness + parent1_second_half_fitness

        if child1_fitness == None: # child2_fitness will always be None if child1_fitness is None
          child1_fitness = parent1.fitness
          child2_fitness = parent2.fitness

        child1 = Chromosome(child1_genetic_code, child1_fitness)
        child2 = Chromosome(child2_genetic_code, child2_fitness)

        self.best_chromosome_fix(removed_nodes1, crossover_point, mutation_point1, child1)
        self.best_chromosome_fix(removed_nodes2, crossover_point, mutation_point2, child2)
       
        return (child1, child2)
    
    def calculate_num_of_ones_limit(self):
      empty_chromosome = Chromosome(0, 0)
      self.better_chromosome_fix(empty_chromosome)
      return empty_chromosome.fitness

    def get_random_position(self, probabilities):
      # first node is chosen if value is in (half-closed) interval [0, probabilities[0])
      # second if it is in interval [probabilites[0], probabilities[1])
      # etc until last node
      # last node is chosen if value is in interval [probabilities[-2], probabilities[-1])
      value = random.randrange(probabilities[-1])

      left = 0
      right = len(self.incoming_neighbors) - 1
      middle = 0
        
      while left <= right:
        middle = (left + right) // 2

        if value < probabilities[0]:
          return self.n  - 1
        if value >= probabilities[middle - 1] and value < probabilities[middle]:
          return self.n - middle - 1
        if probabilities[middle] <= value:
          left = middle + 1
        else:
          right = middle - 1
        
    def initial_population(self):
      population = []
      num_of_ones_limit = self.calculate_num_of_ones_limit()
      # probabilities are set according to in-degree of nodes
      probabilities = [len(self.incoming_neighbors[i]) + sum([len(self.incoming_neighbors[j]) for j in range(i)]) for i in range(self.n)]
      
      while len(population) < self.population_size:
        genetic_code = 0
        for _ in range(num_of_ones_limit):
          index = self.get_random_position(probabilities)
          genetic_code |= (1 << index)

        # we might have duplicate indexes, so fitness is calculated explicitly
        fitness = self.calculate_fitness(genetic_code)
        new_chromosome = Chromosome(genetic_code, fitness)
        self.better_chromosome_fix(new_chromosome)

        population.append(new_chromosome)

      return population

    def tournament_selection_max_diversity(self, population):
      selected = []
      # set of unique chromosomes that are selected
      selected_unique = set({})
      # set of unique chromosomes across whole population
      unique_chromosomes = set(population)

      if len(unique_chromosomes) < self.reproduction_size:
        selected = list(unique_chromosomes)

      while len(selected) < self.reproduction_size:
        candidates = random.sample(population, self.tournament_size)
        if len(selected) < len(unique_chromosomes):
          candidates.sort(key = lambda x: x.fitness)
          for chromosome in candidates:
            if chromosome not in selected_unique:
              selected_unique.add(chromosome)
              selected.append(chromosome)
              break
        else:
          winner = min(candidates, key = lambda x: x.fitness)
          selected.append(winner)

      return selected

    def tournament_selection(self, population):
      selected = []

      for i in range(self.reproduction_size):
        candidates = random.sample(population, self.tournament_size)
        winner = min(candidates, key = lambda x: x.fitness)
        selected.append(winner)

      return selected

    def roulette_wheel_selection(self, chromosomes):
      selected = []
      total_fitness = sum([self.n - chromosome.fitness + 1 for chromosome in chromosomes])
      
      for _ in range(self.reproduction_size):
        value = random.randrange(total_fitness)
      
        current_sum = 0
        for i in range(self.population_size):
          current_sum += (self.n - chromosomes[i].fitness + 1)
          if current_sum > value:
            selected.append(chromosomes[i])
            break
      return selected

    def crossover(self, parent1, parent2):
      # no point in performing crossover on two identical chromosomes
      if parent1 == parent2:
        return (parent1, parent2, None)

      crossover_point = random.randrange(1, self.n)
      child1 = (parent1 >> self.n - crossover_point) << self.n - crossover_point | parent2 & (1 << self.n - crossover_point) - 1
      child2 = (parent2 >> self.n - crossover_point) << self.n - crossover_point | parent1 & (1 << self.n - crossover_point) - 1

      return (child1, child2, crossover_point)

    def uniform_crossover(self, parent1, parent2):
      if parent1 == parent2:
        return parent1, parent2
      
      probability = 0.5
      child1 = child2 = 0

      for i in range(self.n):
        parent1_i_bit = (parent1 >> i) % 2
        parent2_i_bit = (parent2 >> i) % 2
        if random.random() < probability:
          # first child gets i-th allel from first parent, second from second
          child1 |= (parent1_i_bit << i)
          child2 |= (parent2_i_bit << i)
        else:
          # now the situation is reversed, first child gets allel from second parent
          child1 |= (parent2_i_bit << i)
          child2 |= (parent1_i_bit << i)
      
      return (child1, child2)

    def mutation(self, genetic_code):
      value = random.random()
        
      if value < self.mutation_rate:
        mutation_point = random.randrange(self.n)
        helper_number = 1 << (self.n - mutation_point - 1)
        if genetic_code & helper_number == 0:
          return genetic_code | helper_number, self.n - mutation_point - 1
        else:
          return genetic_code - helper_number, self.n - mutation_point - 1

      return genetic_code, None
      
    def generate_new_population(self, reproduction_set, old_population):
      population = []
        
      # old population is already sorted by fitness
      for i in range(self.elitism_size):
        population.append(old_population[i])

      while len(population) < self.population_size:

        index1 = random.randrange(self.reproduction_size)
        index2 = random.randrange(self.reproduction_size)

        parent1 = reproduction_set[index1]
        parent2 = reproduction_set[index2]

        (child1, child2) = self.create_children(parent1, parent2)

        if self.n < 1000:
          population.append(child1)
          population.append(child2)
        
        else:
          if child1 not in population:	
            population.append(child1)	
          if child2 not in population:	
            population.append(child2)	

      return population

    def start(self):
      # trivial cases
      if self.n == 0 or self.n == 1:
        return Chromosome(0, self.n)

      population = self.initial_population()
      population.sort(key = lambda x: x.fitness)

      for i in range(self.iterations):
        if self.n < 1000:
          reproduction_set = self.tournament_selection_max_diversity(population)
        else:
          reproduction_set = self.tournament_selection(population)
        population = self.generate_new_population(reproduction_set, population)

        if i == 0 or i == self.iterations // 2:
          population.sort(key = lambda x: x.fitness)
          population = self.local_search(population)

        population.sort(key = lambda x: x.fitness)

      population = self.local_search(population)
      best_solution = min(population, key = lambda x: x.fitness)
        
      return best_solution

    def local_search(self, population):
      unique_chromosomes = list(set(population))
      limit = len(unique_chromosomes)

      for i in range(limit):
        current_genetic_code = unique_chromosomes[i].genetic_code
        while current_genetic_code:
          new_current_genetic_code = current_genetic_code & (current_genetic_code - 1)
          index = round(math.log2(current_genetic_code - new_current_genetic_code))
          mask = ~(1 << index)
          new_genetic_code = unique_chromosomes[i].genetic_code & mask

          if self.valid_solution(new_genetic_code, index):
            unique_chromosomes[i].genetic_code = new_genetic_code
            unique_chromosomes[i].fitness -= 1

          current_genetic_code = new_current_genetic_code
      
      return population

def check_solution(solution, graph):
  for node in graph:
    if solution & node == 0:
      return False
  return True
