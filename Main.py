import math
def buildGraphWeighted(file):
    # Builds a weighted, undirected graph
    graph = {}
    for line in file:
        v1, v2, w = line.split(',')
        v1, v2 = v1.strip(), v2.strip()
        w = float(w.strip())
        if v1 not in graph:
            graph[v1] = []
        if v2 not in graph:
            graph[v2] = []
        graph[v1].append((v2, w))
        graph[v2].append((v1, w))
    return graph


class Graph:
    # example of adjacency list (or rather map with edge cost i.e. g.

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def getNeighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n):
        H = cityHeuristic
        return H[n]

    def aStarAlgorithm(self, start_node, stop_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}
        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node
        while len(open_list) > 0:
            n = None
            # find a node with the lowest value of f() - evaluation functio
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;
            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_nod
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]
                reconst_path.append(start_node)
                reconst_path.reverse()
                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.getNeighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

cityfile = open("location.txt", "r")
cities = cityfile.readlines()
start = ""
cityHeuristic = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0, "I": 0, "J": 0, "K": 0, "L": 0, "M": 0, "N": 0, "O": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0, "Z": 0}
#create dictionary for heuristics
startFound = False
while startFound != True:
    start = input("input your start position:")
    start = start.upper()
    for x in cities:
        city= x.split(";")
        # print(city[0])
        if start == city[0]:


            startFound =True
            break

targetFound = False
while targetFound != True:
    target = input("input your target position:")
    target = target.upper()
    for x in cities:
        city= x.split(";")
        # print(city[0])
        if target == city[0]:
            L1x = city[1]
            L1y = city[2]
            L1y = L1y.replace("\n", "")

            targetFound =True
            break

for startCity in cities:
    startCity = startCity.replace("\n","")
    citySplit = startCity.split(";")


    L2x = citySplit[1]
    L2y = citySplit[2]

    Dx = (float(L2x) - float(L1x))
    Dy = (float(L2y) - float(L1y))
    distance = math.sqrt((Dx * Dx)+(Dy * Dy))
    cityHeuristic[citySplit[0]] = distance
    # print("Distance to ", citySplit[0], "is ", cityHeuristic[citySplit[0]])

linked = open("Cities.txt", "r")
connectedCities = buildGraphWeighted(linked)
graph1 = Graph(connectedCities)
graph1.aStarAlgorithm(start, target)