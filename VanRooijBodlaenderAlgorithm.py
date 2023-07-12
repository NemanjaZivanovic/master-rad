import math
import copy
import blossom

def calculate_cardinality(set):
    cardinality = 0

    while set:
      set &= (set - 1)  
      cardinality += 1

    return cardinality

def get_maximum_cardinality_set(multiset):
  max_cardinality = -1
  max_cardinality_set = None

  for set in multiset:
    cardinality = calculate_cardinality(set)
    if max_cardinality < cardinality:
      max_cardinality = cardinality
      max_cardinality_set = set

  return max_cardinality_set, max_cardinality

def find_limit(set):
  return math.floor(math.log2(set)) + 1

def find_subset(multiset):
  n = len(multiset)
  
  for i in range(n):
    for j in range(i + 1, n):
      res = multiset[i] & multiset[j]
      if res == multiset[i]:
        return multiset[i]
      elif res == multiset[j]:
        return multiset[j]

  return None

def set_with_frequency_one_element(multiset):
  k = find_limit(max(multiset))

  for i in range(k):
    element = 1 << i
    frequency = 0
    unique = None
    for set in multiset:
      if element & set:
        if frequency == 0:
          frequency = 1
          unique = set
        else:
          frequency = 2
          break
    if frequency == 1:
      return unique
      
  return None

def set_of_sets(multiset, element):
  sets = []

  for i in range(len(multiset)):
    if multiset[i] & element != 0:
      sets.append(i)

  return sets

def subsumption(multiset):
  # Reduction Rule 5
  # if the set of sets containing element e1 is a subset of the set of sets containing element e2 (i.e. e1 is subsumed by e2),
  # any set of sets that covers e1 will always cover e2 also, therefore we can safely remove e2 from the instance.
  k = find_limit(max(multiset))

  set_of_sets_list = [None for _ in range(k)]

  for i in range(k):
    e1 = 1 << i
    if set_of_sets_list[i] == None:
      set_of_sets_list[i] = set_of_sets(multiset, e1)
    se1 = set_of_sets_list[i]
    if len(se1) == 0:
      continue
    for j in range(i + 1, k):
      e2 = 1 << j
      if set_of_sets_list[j] == None:
        set_of_sets_list[j] = set_of_sets(multiset, e2)
      se2 = set_of_sets_list[j]
      if len(se2) == 0:
        continue
      if set(se1) <= set(se2):
        return e2, None
      elif set(se2) <= set(se1):
        return e1, None

  return None, set_of_sets_list

def calculate_union_from_reduction_rule6(multiset, set, frequency_two_elements, set_of_sets_list):
  union = 0
  
  for element in frequency_two_elements:
    se = set_of_sets_list[element]
    for i in se:
      union |= multiset[i]

  if union != 0:
    return union - set

  return union

def calculate_frequency_two_elements_in_set(multiset, set):
  frequency_two_elements = []

  help_set = set
  indexes = []
  while help_set:
    new_help_set = help_set & (help_set - 1)
    index_el = round(math.log(help_set - new_help_set, 2))
    indexes.append(index_el)
    help_set = new_help_set
  
  for index in indexes:
    element = 1 << index
    frequency = 0
    for j in range(len(multiset)):
      if element & multiset[j]:
        frequency += 1
        if frequency > 2:
          break
    if frequency == 2:
      frequency_two_elements.append(index)

  return frequency_two_elements

def counting_arguments(multiset, set_of_sets_list):
  # Reduction Rule 6
  # For any set R with r2 elements of frequency two, we let q be the number of elements 
  # in the sets containing a frequency two element from R that are not in R themselves.
  # If q < r2, taking R is always as least as good as discarding R since then we use r2 − 1 
  # sets less while also covering q < r2 less elements we can always cover these elements by 
  # picking one additional set per element.
  frequency_two_elements_dict = dict()

  for set in multiset:
    frequency_two_elements = calculate_frequency_two_elements_in_set(multiset, set)
    frequency_two_elements_dict[set] = frequency_two_elements
    q = calculate_union_from_reduction_rule6(multiset, set, frequency_two_elements, set_of_sets_list)
    r2 = len(frequency_two_elements)
    if calculate_cardinality(q) < r2:
      return set, None

  return None, frequency_two_elements_dict

def remove_set_of_size_two_with_only_frequency_two_elements(multiset, frequency_two_elements_dict, set_of_sets_list):
  # Reduction Rule 7
  # If there exists a set R of cardinality two containing two frequency two elements e1, e2, 
  # such that ei occurs in R and Ri, then the reduction rule transforms this instance 
  # into an instance where R, R1 and R2 have been replaced by the set Q = (R1 ∪ R2) \ R.
  # There exist a minimum set cover of (S, U) that either contains R, or contains both R1 and R2. 
  # This is so because if we take only one set from R, R1 and R2, then this must be R since we must cover e1 and e2; 
  # if we take two, then it is of no use to take R since the other two cover more elements; 
  # and, if we take all three, then the set cover is not minimal.
  # The rule postpones the choice between the first two possibilities, taking Q in the minimum set cover
  # of the transformed problem if both R1 and R2 are in a minimum set cover, or taking no set in the
  # minimum set cover of the transformed problem if R is in a minimum set cover.

  for set in multiset:
    # next two lines are the same as expression `if calculate_cardinality(set) == 2`, only faster
    # because we don't actually need calculate all of the cardinality of set, only to check if it's
    # cardinality is equal to 2
    temp = set & (set - 1)
    if temp & (temp - 1) == 0:
      #frequency_two_elements = calculate_frequency_two_elements_in_set(multiset, set)
      frequency_two_elements = frequency_two_elements_dict[set]
      if len(frequency_two_elements) == 2:
        Q = calculate_union_from_reduction_rule6(multiset, set, frequency_two_elements, set_of_sets_list)
        se1 = set_of_sets_list[frequency_two_elements[0]]
        if multiset[se1[0]] == set:
          R1 = multiset[se1[1]]
        else:
          R1 = multiset[se1[0]]
        se2 = set_of_sets_list[frequency_two_elements[1]]
        if multiset[se2[0]] == set:
          R2 = multiset[se2[1]]
        else:
          R2 = multiset[se2[0]]

        multiset = remove_set(multiset, set)
        multiset = remove_set(multiset, R1)
        multiset = remove_set(multiset, R2)

        multiset.append(Q)
        #multiset = remove_elements(multiset, set)
        C = MSC(multiset)

        #if Q in C:
        #  C.remove(Q)
        #  C.append(R1)
        #  C.append(R2)
        #else:
         # C.append(set)

        has_subset = False
        for el in C:
          if el & Q == el:
            has_subset = True
            C.remove(el)
            C.append(R1)
            C.append(R2)
            break
        if has_subset == False:
          C.append(set)
        return C

  return None

def get_elements_indexes_from_set(set):
  indexes = []

  while(set):
    help_set = set & (set - 1)
    index = round(math.log(set - help_set, 2))
    set = help_set
    indexes.append(index)

  return indexes

def get_set_from_match(index1, index2):
  return (1 << index1) | (1 << index2)

def maximum_matching(multiset):
  graph = blossom.Graph()

  for set in multiset:
    idx1, idx2 = get_elements_indexes_from_set(set)
    if idx1 > idx2:
      idx1, idx2 = idx2, idx1
    graph.add_edge((idx1, idx2))
  
  matching = blossom.Matching()
  matching.add_vertices(graph.get_vertices())
  maximum_matching = blossom.get_maximum_matching(graph, matching)

  R = []

  for edge in maximum_matching.edges:
    R.append(get_set_from_match(edge[0], edge[1]))

  for vertex in maximum_matching.exposed_vertices:
    if vertex in graph.adjacency:
      neighbor = graph.adjacency[vertex].pop()
      R.append(get_set_from_match(vertex, neighbor))

  return R

def remove_elements(multiset, set):
  multiset = copy.copy(multiset)

  for i in range(len(multiset)):
    if multiset[i] & set != 0:
      multiset[i] -= (multiset[i] & set)

  multiset = list(filter(lambda x: x != 0, multiset))
  return multiset

def remove_element(multiset, element):
  return remove_elements(multiset, element)

def remove_set(multiset, set):
  multiset = copy.copy(multiset)
  multiset.remove(set)
  return multiset

def convert_msc_to_mds_solution(multiset, msc_result):
  mds_solution_indexes = []

  for i in range(len(msc_result)):
    for j in range(len(multiset)):
      if msc_result[i] & multiset[j] == msc_result[i]:
          mds_solution_indexes.append(j)
          break

  mds_solution_indexes.sort()
  mds_solution = 0
  for i in mds_solution_indexes:
    mds_solution |= (1 << (len(multiset) - i - 1))

  return mds_solution

def VanRooijBodlaender(graph):
  msc_solution = MSC(graph)
  return len(msc_solution)
  #return convert_msc_to_mds_solution(graph, msc_solution)

def MSC(multiset):
  if len(multiset) == 0:
    return []
  
  set = set_with_frequency_one_element(multiset)
  if set != None:
    multiset = remove_set(multiset, set)
    multiset = remove_elements(multiset, set)
    return [set] + MSC(multiset)
  
  subset = find_subset(multiset)
  if subset != None:
    multiset = remove_set(multiset, subset)
    return MSC(multiset)
  
  e2, set_of_sets_list = subsumption(multiset)
  if e2 != None:
    multiset = remove_element(multiset, e2)
    return MSC(multiset)
  
  set, frequency_two_elements_dict = counting_arguments(multiset, set_of_sets_list)
  if set != None:
    multiset = remove_set(multiset, set)
    multiset = remove_elements(multiset, set)
    return [set] + MSC(multiset)
  
  value = remove_set_of_size_two_with_only_frequency_two_elements(multiset, frequency_two_elements_dict, set_of_sets_list)
  if value != None:
    return value
  
  maximum_cardinality_set, maximum_cardinality = get_maximum_cardinality_set(multiset)
  
  if maximum_cardinality <= 2:
    return maximum_matching(multiset)

  C1 = MSC(remove_set(multiset, maximum_cardinality_set))
  C2 = [maximum_cardinality_set] + MSC(remove_elements(multiset, maximum_cardinality_set))

  return C1 if len(C1) < len(C2) else C2