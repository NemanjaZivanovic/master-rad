import random
import math

def generate_directed_graph(n):
  limit = 2 ** n
  graph = []

  for i in range(n):
    number = random.randrange(limit)
    number |= (1 << n - i - 1)
    graph.append(number)

  return graph

def generate_directed_graph_with_limited_num_of_edges_per_node(n):
  graph = []

  for i in range(n):
    number = 0
    for j in range(random.randrange(round(math.sqrt(n)))):
      number |= (1 << random.randrange(n))
    
    number |= (1 << n - i - 1)
    graph.append(number)

  return graph

def generate_undirected_graph(n):
  graph = []

  for i in range(n):
    number = 0
    for j in range(random.randrange(n)):
      number |= (1 << random.randrange(n))
   
    number |= (1 << n - i - 1)
    graph.append(number)

  for i in range(n):
    for j in range(n):
      if (1 << n - j - 1) & graph[i] != 0:
        graph[j] |= (1 << n - i - 1)

  return graph

def generate_undirected_graph_with_limited_num_of_edges_per_node(n):
  graph = []

  for i in range(n):
    number = 0
    for j in range(random.randrange(round(math.sqrt(n)))):
      number |= (1 << random.randrange(n))
    
    number |= (1 << n - i - 1)
    graph.append(number)

  for i in range(n):
    for j in range(n):
      if (1 << n - j - 1) & graph[i] != 0:
        graph[j] |= (1 << n - i - 1)

  return graph

def print_graph(graph):
  n = len(graph)

  for i in range(n):
    print(f"{i + 1}:", end = ' ')
    print("[ ", end = '')
    for j in range(n):
      if i != j:
        if graph[i] & (1 << n - j - 1):
          print(f"{j + 1}", end = ' ')
    print("]")

def print_solution(solution, n):
  print("[ ", end = '')

  for i in range(n):
    if solution & (1 << n - i - 1):
      print(f"{i + 1}", end = ' ')

  print("]") 

def write_graph_to_file(graph, filename):
  n = len(graph)
  f = open(filename, "w")

  for i in range(n):
    f.write(f"{i + 1}: ")
    f.write("[ ")
    for j in range(n):
      if i != j:
        if graph[i] & (1 << n - j - 1):
          f.write(f"{j + 1} ")
    f.write("]\n")

  f.close()
  
def load_graph_from_file(filename):
  f = open(filename, "r")
  graph = []

  lines = f.readlines()
  n = len(lines)

  for index, row in enumerate(lines):
    neighbors = row.split(" ")[2:-1]
    number = 0
    for i in range(len(neighbors)):
      number |= (1 << n - int(neighbors[i]))
    number |= (1 << (n - index - 1))
    graph.append(number)

  f.close()
  return graph