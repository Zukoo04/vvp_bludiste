import numpy as np
import csv
import matplotlib.pyplot as plt
import heapq
import matplotlib.colors
import os

def load_maze_from_csv(filename):
    """
    Načte bludiště z CSV souboru.

    Args:
    filename (str): Název CSV souboru.

    Returns:
    numpy.ndarray: Matice reprezentující bludiště.
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'r') as file:
        maze_reader = csv.reader(file)
        maze = []
        for row in maze_reader:
            maze.append([1 if cell == '1' else 0 for cell in row])
    return np.array(maze)

def dijkstra_shortest_path(maze):
    """
    Najde nejkratší cestu v bludišti pomocí Dijkstrova algoritmu.

    Args:
    maze (numpy.ndarray): Matice reprezentující bludiště.

    Returns:
    list of tuples: Cesty od startu do cíle.
    """
    n = len(maze)
    start = (0, 0)
    end = (n - 1, n - 1)
    distances = {(i, j): float('inf') for i in range(n) for j in range(n)}
    distances[start] = 0
    visited = set()
    prev = {}
    queue = [(0, start)]

    while queue:
        current_dist, current_node = heapq.heappop(queue)
        if current_node == end:
            path = []
            while current_node in prev:
                path.append(current_node)
                current_node = prev[current_node]
            path.append(start)
            return path[::-1]

        visited.add(current_node)
        row, col = current_node
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        for neighbor_row, neighbor_col in neighbors:
            if 0 <= neighbor_row < n and 0 <= neighbor_col < n:
                neighbor_node = (neighbor_row, neighbor_col)
                if maze[neighbor_row][neighbor_col] == 0 and neighbor_node not in visited:
                    new_dist = current_dist + 1
                    if new_dist < distances[neighbor_node]:
                        distances[neighbor_node] = new_dist
                        prev[neighbor_node] = current_node
                        heapq.heappush(queue, (new_dist, neighbor_node))
    return []

def draw_maze_with_path(maze, path):
    """
    Vykreslí bludiště s nalezenou cestou.

    Args:
    maze (numpy.ndarray): Matice reprezentující bludiště.
    path (list of tuples): Cesty od startu do cíle.
    """
    maze_with_path = maze.copy()
    for node in path:
        maze_with_path[node[0]][node[1]] = 2

    cmap = matplotlib.colors.ListedColormap(['white', 'black', 'red'])

    plt.imshow(maze_with_path, cmap=cmap)
    plt.show()
    
def incidence_matrix(maze):
    """
    Vytvoří incidenční matici bludiště.

    Args:
    maze (numpy.ndarray): Matice reprezentující bludiště.

    Returns:
    numpy.ndarray: Incidenční matice bludiště.
    """
    n = len(maze)
    incidence = np.zeros((n * n, n * n), dtype=int)
    for i in range(n):
        for j in range(n):
            if not maze[i][j]:
                index = i * n + j
                if i > 0 and not maze[i - 1][j]:
                    incidence[index][index - n] = 1
                if i < n - 1 and not maze[i + 1][j]:
                    incidence[index][index + n] = 1
                if j > 0 and not maze[i][j - 1]:
                    incidence[index][index - 1] = 1
                if j < n - 1 and not maze[i][j + 1]:  
                    incidence[index][index + 1] = 1
    return incidence

def generate_maze(size, template="empty", barrier_density=0.2):
    """
    Generuje bludiště dané velikosti a šablonou.

    Args:
    size (int): Velikost bludiště (počet řádků a sloupců).
    template (str): Šablona bludiště (možné hodnoty: "empty", "slalom", atd.).
    barrier_density (float): Hustota překážek v bludišti.

    Returns:
    numpy.ndarray: Matice reprezentující vygenerované bludiště.
    """
    if template == "empty":
        maze = np.zeros((size, size), dtype=int)
    elif template == "random":
        maze = np.random.choice([0, 1], size=(size, size), p=[1 - barrier_density, barrier_density])
    elif template == "slalom":
        maze = np.zeros((size, size), dtype=int)
        for i in range(1, size, 4):
            for j in range(size):
                maze[i][j] = 1
        for i in range(3, size, 4):
            for j in range(size):
                maze[i][j] = 1
        # Ensure there is a path from the top left to the bottom right
        for i in range(1, size-1, 4):
            maze[i+1][np.random.randint(0, size)] = 0
            maze[i][np.random.randint(0, size)] = 0 # Make slalom passable

        # Make sure there is a clear path from top left to bottom right
        for i in range(1, size-1, 2):
            if i % 4 == 1:
                maze[i][size-1] = 0
            else:
                maze[i][0] = 0
        # Ensure there is a zero in the bottom right corner
        maze[size-1][size-1] = 0
    else:
        raise ValueError("Neznámá šablona bludiště.")
    return maze
# Testování funkcí
maze = load_maze_from_csv("maze_5.csv")
incidence = incidence_matrix(maze)
shortest_path = dijkstra_shortest_path(maze)
draw_maze_with_path(maze, shortest_path)
