import itertools

def count_number_of_binary_ones(number):
  ones = 0
  while number:
    number &= (number - 1)
    ones += 1
  return ones

def check_solution(number, graph):
  for node in graph:
    if number & node == 0:
      return False
  return True

def get_transpose_graph(graph):
    transpose_graph = []
    n = len(graph)

    for i in range(n):
        set = 0
        for j in range(n):
            if graph[j] & (1 << n - i - 1):
                set |= (1 << n - j - 1)
        transpose_graph.append(set)

    return transpose_graph

def brute_force_algorithm(graph):
  graph = get_transpose_graph(graph)
  solution = []
  min_solution_fitness = float('inf')
  limit = 2 ** len(graph)

  for number in range(limit):
    fitness = count_number_of_binary_ones(number)
    if fitness < min_solution_fitness and check_solution(number, graph):
      solution = number
      min_solution_fitness = fitness

  return solution

def enhanced_brute_force_algorithm_itertools(graph):
  n = len(graph)
  graph = get_transpose_graph(graph)
  for length in range(1, n + 1):
    for subset in itertools.combinations(range(1, n + 1), length):
      current_value = 0
      for el in subset:
        current_value |= (1 << (n - el))
      if check_solution(current_value, graph):
        return current_value

def enhanced_brute_force_algorithm(graph):
  n = len(graph)
  graph = get_transpose_graph(graph)
  starting_value = 0
  for length in range(n + 1):

    # 00001 -> 00011 -> 00111 -> 01111 -> 11111 for n = 5
    starting_value = (starting_value << 1) + 1
    current_value = starting_value
    # 10000 -> 11000 -> 11100 -> 11110 -> 11111 for n = 5
    end_value = (1 << n - length) * starting_value

    while current_value <= end_value:
      # idea for generating the next combination of size `length` was taken from
      # https://www.geeksforgeeks.org/next-higher-number-with-same-number-of-set-bits/
      next = 0
      rightOne = current_value & -(current_value)
      nextHigherOneBit = current_value + int(rightOne)
      rightOnesPattern = current_value ^ int(nextHigherOneBit)
      rightOnesPattern = (int(rightOnesPattern) /
                          int(rightOne))
      rightOnesPattern = int(rightOnesPattern) >> 2
      next = nextHigherOneBit | rightOnesPattern
      current_value = next

      if check_solution(current_value, graph):
        return current_value