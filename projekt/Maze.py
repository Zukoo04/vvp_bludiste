import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import csv
import heapq
from scipy.sparse import lil_matrix, csr_matrix
import random

class Maze:
    def __init__(self, size: int = None, template: str = "empty", barrier_density: float = 0.2, filename: str = None) -> None:
        if filename:
            self.maze = self.load_maze_from_csv(filename)
        else:
            self.maze = self.generate_maze(size, template, barrier_density)
        self.size = self.maze.shape[0]
        self.shortest_path = None
        self.incidence = None
    
    @staticmethod
    def load_maze_from_csv(filename: str) -> np.ndarray:
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
    
    def generate_maze(self, size: int, template: str = "empty", barrier_density: float = 0.2) -> np.ndarray:
        """
        Generuje bludiště dané velikosti a šablonou s garantovanou cestou od (0, 0) do (size-1, size-1).

        Args:
        size (int): Velikost bludiště (počet řádků a sloupců).
        template (str): Šablona bludiště (možné hodnoty: "empty", "slalom", "random").
        barrier_density (float): Hustota překážek v bludišti.

        Returns:
        numpy.ndarray: Matice reprezentující vygenerované bludiště.
        """
        if template == "empty":
            maze = np.zeros((size, size), dtype=int)
        elif template == "random":
            while True:
                maze = np.random.choice([0, 1], size=(size, size), p=[1 - barrier_density, barrier_density])
                # Ensure there is a path from top-left to bottom-right
                if self.is_path_exists((0, 0), (size-1, size-1), maze):
                    break
        elif template == "slalom":
            maze = np.zeros((size, size), dtype=int)
            for i in range(1, size, 4):
                for j in range(size):
                    maze[i][j] = 1
            for i in range(3, size, 4):
                for j in range(size):
                    maze[i][j] = 1
            for i in range(1, size-1, 4):
                maze[i+1][np.random.randint(0, size)] = 0
                maze[i][np.random.randint(0, size)] = 0  # Make slalom passable
            for i in range(1, size-1, 2):
                if i % 4 == 1:
                    maze[i][size-1] = 0
                else:
                    maze[i][0] = 0
            maze[size-1][size-1] = 0
        else:
            raise ValueError("Neznámá šablona bludiště.")
        
        return maze
    
    def is_path_exists(self, start, end, maze):
        """
        Kontroluje, zda existuje cesta mezi startem a koncem v daném bludišti.
        
        Args:
        start (tuple): Pozice začátku cesty (řádek, sloupec).
        end (tuple): Pozice konce cesty (řádek, sloupec).
        maze (numpy.ndarray): Matice bludiště.

        Returns:
        bool: True, pokud cesta existuje, jinak False.
        """
        n = maze.shape[0]
        visited = set()
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            if current == end:
                return True
            visited.add(current)
            row, col = current
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            for neighbor in neighbors:
                n_row, n_col = neighbor
                if 0 <= n_row < n and 0 <= n_col < n and maze[n_row][n_col] == 0 and neighbor not in visited:
                    queue.append(neighbor)
        
        return False
    
    def dijkstra_shortest_path(self) -> list:
        """
        Najde nejkratší cestu v bludišti pomocí Dijkstrova algoritmu.

        Returns:
        list of tuples: Cesty od startu do cíle.
        """
        n = self.size
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
                self.shortest_path = path[::-1]
                return self.shortest_path
    
            visited.add(current_node)
            row, col = current_node
            neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
            for neighbor_row, neighbor_col in neighbors:
                if 0 <= neighbor_row < n and 0 <= neighbor_col < n:
                    neighbor_node = (neighbor_row, neighbor_col)
                    if self.maze[neighbor_row][neighbor_col] == 0 and neighbor_node not in visited:
                        new_dist = current_dist + 1
                        if new_dist < distances[neighbor_node]:
                            distances[neighbor_node] = new_dist
                            prev[neighbor_node] = current_node
                            heapq.heappush(queue, (new_dist, neighbor_node))
        return []
    
    def draw_maze_with_path(self) -> None:
        """
        Vykreslí bludiště s nalezenou cestou.
        """
        if self.shortest_path is None:
            raise ValueError("Nejprve najděte nejkratší cestu pomocí dijkstra_shortest_path.")
    
        maze_with_path = self.maze.copy()
        for node in self.shortest_path:
            maze_with_path[node[0]][node[1]] = 2
    
        cmap = matplotlib.colors.ListedColormap(['white', 'black', 'red'])
    
        plt.imshow(maze_with_path, cmap=cmap)
        plt.show()
