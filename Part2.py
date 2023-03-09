import argparse, heapq, math, random, time, sys
from typing import Tuple, Dict, List

import stubs.pickle_file

from collections import deque

CITY_LOCATIONS = {
    "Vancouver": (49.2827, -123.1207),
    "North Vancouver": (49.3165, -123.0720),
    "West Vancouver": (49.3730, -123.2900),
    "Burnaby": (49.2488, -122.9805),
    "Coquitlam": (49.2827, -122.7912),
    "Delta": (49.0847, -123.0587),
    "Richmond": (49.1632, -123.1376),
    "Surrey": (49.1913, -122.8490),
    "Langley": (49.1044, -122.6606),
    "Abbotsford": (49.0504, -122.3045),
    "Mission": (49.1337, -122.3095),
    "Chilliwack": (49.1579, -121.9514),
    "New Westminster": (49.2070, -122.9110),
}


# Grassfire search algorithm
def grassfire_search(graph: Dict[str, Dict[str, float]], start: str, goal: str, **kwargs) -> Tuple[list, str]:
    """
    Find the shortest path between two nodes in a weighted graph using grassfire algorithm

    :param graph: A dictionary representing the graph, where each key is a node and its value is a dictionary of
        neighboring nodes and their edge weights.
    :param start: The node to start the search from.
    :param goal: The node to search for.
    :return: A tuple containing a list of nodes in the shortest path from start to goal, and a string representing
        the total cost of the path (in kilometers) rounded to 2 decimal places.
    """
    queue = deque()
    queue.append(start)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for next in graph[current]:
            new_cost = cost_so_far[current] + graph[current][next]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                queue.append(next)
                came_from[next] = current

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()

    return path, f'{cost_so_far[goal] / 1000:.2f}'


def minimum_distance(city1: str, city2: str, city_locations: Dict[str, Tuple[float, float]]) -> float:
    """
    Calculate the Euclidean distance between two cities represented as (x, y) coordinates.

    :param city1: A string representing the name of the first city.
    :param city2: A string representing the name of the second city.
    :param city_locations: A dictionary representing the (x, y) coordinates of all cities, where each key is the
        name of a city and its value is a tuple of two floats representing the x and y coordinates.
    :return: A float representing the Euclidean distance between the two cities.
    """
    x1, y1 = city_locations[city1]
    x2, y2 = city_locations[city2]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def get_successors(city: str, city_locations: Dict[str, Tuple[float, float]]) -> List[str]:
    """
    Get a list of all successor nodes of a given node.

    :param city: A string representing the name of the node to get successors for.
    :param city_locations: A dictionary representing the (x, y) coordinates of all nodes, where each key is the
        name of a node and its value is a tuple of two floats representing the x and y coordinates.
    :return: A list of strings representing the names of the successor nodes.
    """
    successors = []
    for successor in city_locations.keys():
        if successor != city:
            successors.append(successor)
    return successors


def heuristic(city: str, goal: str, city_locations: Dict[str, Tuple[float, float]]) -> int:
    """
    Calculate the heuristic value for a city using the minimum distance to the goal and a random offset.

    :param city: A string representing the name of the city to calculate the heuristic for.
    :param goal: A string representing the name of the goal city.
    :param city_locations: A dictionary representing the (x, y) coordinates of all cities, where each key is the
        name of a city and its value is a tuple of two floats representing the x and y coordinates.
    :return: An integer representing the heuristic value for the city.
    """
    x = min([minimum_distance(city, goal, city_locations) for city in city_locations])
    y = random.randint(5, 10)
    return math.floor(x - y)


def heuristic_consistent(city: str, goal: str, city_locations: Dict[str, Tuple[float, float]]) -> int:
    """
    Calculate the heuristic value for a city using the minimum distance to the goal and a random offset.
    Ensure the heuristic is consistent by adjusting the random offset for each node.

    :param city: A string representing the name of the city to calculate the heuristic for.
    :param goal: A string representing the name of the goal city.
    :param city_locations: A dictionary representing the (x, y) coordinates of all cities, where each key is the
        name of a city and its value is a tuple of two floats representing the x and y coordinates.
    :return: An integer representing the heuristic value for the city.
    """
    x = min([minimum_distance(city, goal, city_locations) for city in city_locations])
    y = random.randint(5, 10)
    for successor in get_successors(city, city_locations):
        successor_distance = minimum_distance(successor, goal, city_locations)
        movement_cost = math.sqrt((city_locations[city][0] - city_locations[successor][0]) ** 2
                                  + (city_locations[city][1] - city_locations[successor][1]) ** 2)
        max_movement_cost = movement_cost + successor_distance
        y = min(y, max(x - successor_distance, max_movement_cost - x))
    return math.floor(x - y)


def astar_search(graph: Dict[str, Dict[str, float]], start: str, goal: str,
                 **kwargs: Dict[str, Tuple[float, float]]) -> Tuple[List[str], str]:
    """Find the shortest path from the start to the goal node using A* algorithm.

    :param graph: A dictionary representing the graph, where each key is a node and its value is a dictionary of
        neighboring nodes and their edge weights.
    :param start: The node to start the search from.
    :param goal: The node to search for.
    :param kwargs: A dictionary that contains additional keyword arguments. Here it should contain 'city_locations', a
                   dictionary that maps each city to its x and y coordinates.
    :return: A tuple containing the path from start to end nodes and the distance to travel that path.
    """
    city_locations = kwargs['city_locations']
    heur_func = kwargs['heuristic_func']

    # The frontier is a list that holds the nodes to be explored
    # We then initialize the heap with the start and distance at 0
    # came_from is a dictionary that maps each explored node to the
    # node that came before it in the shortest path. cost_so_far is
    # a dictionary that maps each node to the cost of the shortest
    # path from the starting node to that node. The dictionaries are
    # initialized with the starting node.
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    # This loop runs until the frontier list is empty or the goal node is reached.
    # In each iteration, the node with the lowest cost is removed from the frontier
    # list and assigned to current. If current is the goal node, the loop is exited.
    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        # This loop examines each neighbor of the current node.
        # For each neighbor next, it computes the cost of the path
        # from the starting node to next through current. If this
        # cost is lower than the previously computed cost (or next
        # has not been explored yet), cost_so_far is updated. The
        # priority of the next node is set as the sum of the actual
        # cost and the heuristic value of the estimated remaining cost.
        # The node is then added to the frontier list and its came_from value is updated.
        for next in graph[current]:
            new_cost = cost_so_far[current] + graph[current][next]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                if heur_func:
                    priority = new_cost + heuristic_consistent(next, goal,
                                                               city_locations)  # use actual cost + heuristic
                else:
                    priority = new_cost + heuristic(next, goal, city_locations)  # use actual cost + heuristic
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    # Backtrack to construct the path used from end->start // start->end
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()

    return path, f'{cost_so_far[goal] / 1000:.2f}'


def dijkstra_search(graph: dict, start: str, end: str, **kwargs: dict) -> tuple:
    """
    Searches for the shortest path between two nodes in a graph using Dijkstra's algorithm.

    :param graph: A dictionary representation of the graph. The keys are nodes, and the values are dictionaries
                  that represent the edges between the node and its neighbors. Each neighbor is a key in the
                  inner dictionary, and the value is the weight of the edge between the node and its neighbor.
    :param start: The starting node for the search.
    :param end: The target node for the search.
    :param kwargs: A dictionary containing keyword arguments. Currently unused.
    :return: A tuple containing two elements: a list of nodes in the shortest path from the starting node to the
             target node, and a string representing the total weight of the path in kilometers.
    """
    pq = []  # priority queue to store nodes with the lowest cost
    visited = set()  # set of visited nodes
    dist = {start: 0}  # dictionary to keep track of the distance from the start node to each node
    prev = {}  # dictionary to keep track of the previous node for each node

    heapq.heappush(pq, (0, start))

    while pq:
        (cost, node) = heapq.heappop(pq)
        if node == end:
            path = [node]
            while node != start:
                node = prev[node]
                path.append(node)
            path.reverse()
            return path, f'{cost / 1000:.2f}'

        if node not in visited:
            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_cost = cost + graph[node][neighbor]
                    if neighbor not in dist or new_cost < dist[neighbor]:
                        heapq.heappush(pq, (new_cost, neighbor))
                        dist[neighbor] = new_cost
                        prev[neighbor] = node

    return None, None


def main(args):
    """
    Main flow of the program, correct names of cities,
    get cost and path, return to user.

    :param args: argparse Namespace object
    """
    ALGORITHMS = {
        'A*': astar_search,
        'Dijkstra': dijkstra_search,
        'Grassfire': grassfire_search
    }

    print(f'Running {args.Algorithm} ...')

    start_city = args.City_1.lower()
    end_city = args.City_2.lower()
    start_city = start_city.title()
    end_city = end_city.title()

    # Execute the algorithm as given via cmdline
    path, cost = ALGORITHMS[args.Algorithm](CONNECTIVITY_MAP, start_city, end_city, city_locations=CITY_LOCATIONS,
                                            heuristic_func=args.consistent)

    # Print the results
    if path is not None:
        print(f"Shortest path from {args.City_1} to {args.City_2}:\n    {' -> '.join(path)}")
        print(f"    Total cost: {cost}Km")
    else:
        print(f"No path found from {args.City_1} to {args.City_2}")


if __name__ == '__main__':
    CONNECTIVITY_MAP = stubs.pickle_file.load_data('resources/distances.pickle')

    start_time = time.time()
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('Required')
    required.add_argument('City_1', help='Input name for starting city')
    required.add_argument('City_2', help='Input name for ending city')
    required.add_argument('Algorithm', choices=['A*', 'Grassfire', 'Dijkstra'], help='Algorithm Choice')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--consistent', default=False, action='store_true',
                          help='Set this flag if you`d like to use consistent heuristics')

    arguments = parser.parse_args()

    if arguments.City_1 not in CITY_LOCATIONS:
        print(f'Please check the spelling of this city: {arguments.City_1}', file=sys.stderr)
        sys.exit(1)
    elif arguments.City_2 not in CITY_LOCATIONS:
        print(f'Please check the spelling of this city: {arguments.City_2}', file=sys.stderr)
        sys.exit(2)
    else:
        main(arguments)
        end_time = time.time()
        print(f'Total Execution time: {(end_time - start_time):2f} Seconds.')
