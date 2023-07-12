from irace import irace
import Graph, MDSGeneticAlgorithm
import time, os

def target_runner(experiment, scenario):
    global graphs, n, num_of_calls
    i = experiment['instance']
    graph = graphs[i]
    seed = experiment['seed']
    iterations = int(experiment['configuration']['iterations'])
    population_size = int(experiment['configuration']['population_size'])
    reproduction_size = int(experiment['configuration']['reproduction_size'])
    tournament_size = int(experiment['configuration']['tournament_size'])
    mutation_rate = 0.5
    elitism_size = int(experiment['configuration']['elitism_size'])
    genetic_algorithm = MDSGeneticAlgorithm.MDSGeneticAlgorithm(graph, seed, iterations, 
                                                                population_size, reproduction_size,
                                                                tournament_size, mutation_rate,
                                                                elitism_size)
    try:
        best_solution = genetic_algorithm.start()
    except:
        return dict(cost=n*10)
    return dict(cost=best_solution.fitness)

parameters_table = '''
iterations        ""    i    (5, 100)
population_size   ""    i    (1, 50)
reproduction_size ""    i    (1, 20)
tournament_size   ""    i    (2, 5)
elitism_size      ""    i    (1, 5)
'''

default_values = '''
iterations  population_size reproduction_size tournament_size elitism_size
5           1              1                 3                1 
'''


num_of_graphs = 100
n = 10
node_number_string = f"graphs with {n} nodes"
graphs_folder = "graphs_folder"
graphs = []
for i in range(num_of_graphs):
    path = os.path.join(graphs_folder, node_number_string, f"graph_num_{i + 1}_with_{n}_nodes.txt")
    graph = Graph.load_graph_from_file(path)
    graphs.append(graph)

instances = range(num_of_graphs)
#boundMax = 5

# See https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
scenario = dict(
    instances = instances,
    maxExperiments = 5000,
    debugLevel = 1,
    digits = 10,
    parallel = 2 # It can run in parallel ! 
    )

tuner = irace(scenario, parameters_table, target_runner)
tuner.set_initial_from_str(default_values)
best_confs = tuner.run()

print(best_confs)
