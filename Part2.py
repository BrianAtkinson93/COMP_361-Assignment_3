import argparse, heapq, math, random, time, sys
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


# Distance heuristic
def distance(node, goal):
    return ((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2) ** 0.5


# Grassfire search algorithm
def grassfire_search(graph, start, goal, **kwargs):
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


def minimum_distance(city1, city2, city_locations):
    x1, y1 = city_locations[city1]
    x2, y2 = city_locations[city2]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def heuristic(city, goal, city_locations):
    x = min([minimum_distance(city, goal, city_locations) for city in city_locations])
    y = random.randint(5, 10)
    return math.floor(x - y)


def astar_search(graph, start, goal, **kwargs) -> tuple:
    """

    :param graph: A dictionary that represents the weighted graph
    :param start: The starting Node
    :param goal: The Ending Node
    :param kwargs: object containing keyword arguments
    :return: A tuple containing the path from start to end nodes and the distance to travel that path
    """
    city_locations = kwargs['city_locations']

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
                priority = new_cost + heuristic(next, goal, city_locations)  # use actual cost + heuristic
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    # Backtrack to construct the path used from end->start // start->end
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()

    return path, f'{cost_so_far[goal] / 1000:.2f}'


def dijkstra_search(graph, start, end, **kwargs):
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
    ALGORITHMS = {
        'A*': astar_search,
        'Dijkstra': dijkstra_search,
        'Grassfire': grassfire_search
    }

    start_city = args.City_1.lower()
    end_city = args.City_2.lower()
    start_city = start_city.title()
    end_city = end_city.title()

    # Execute the algorithm as given via cmdline
    path, cost = ALGORITHMS[args.Algorithm](CONNECTIVITY_MAP, start_city, end_city, city_locations=CITY_LOCATIONS)

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
    parser.add_argument('City_1', help='Input name for starting city')
    parser.add_argument('City_2', help='Input name for ending city')

    required = parser.add_argument_group('Required')
    required.add_argument('Algorithm', choices=['A*', 'Grassfire', 'Dijkstra'], help='Algorithm Choice')

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
