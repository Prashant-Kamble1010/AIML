# 1.	Implement the Informed Search algorithm for real-life problems.
# 4. Develop a pathfinding solution using the A* algorithm for a maze-based game environment. The agent must find the most cost-efficient route from the start position to the goal, 
# considering movement costs and a suitable heuristic function (e.g., Manhattan distance) to guide the search efficiently.
from queue import PriorityQueue
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0]
]
start = (0, 0)
goal = (3, 4)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def a_star(start, goal):
    pq = PriorityQueue()
    pq.put((0, start, [start]))   # (priority, current_node, path)
    visited = set()               
    
    while not pq.empty():
        cost, node, path = pq.get()
        if node == goal:
            return path        
        x, y = node
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0 and (nx,ny) not in visited:
                g = len(path)        
                h = heuristic((nx, ny), goal)  
                f = g + h
                pq.put((f, (nx, ny), path + [(nx, ny)]))
                visited.add(node)

    return None  
path = a_star(start, goal)
if path:
    print("Shortest Path Found:")
    print(path)
    print(f"Total Steps: {len(path)-1}")
else:
    print("No Path Found.")
Output:
Shortest Path Found:
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3), (3, 4)]
Total Steps: 7



# 2. Design an algorithm using Breadth-First Search (BFS) to find the shortest path from a start node to a goal node in a maze represented as a grid graph. The maze contains obstacles 
# (walls) and free cells. Implement BFS to ensure that the first found path is the optimal one in terms of the number of steps
from queue import Queue
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0]
]
start = (0, 0)
goal = (3, 4)
def bfs(start, goal):
    q = Queue()
    q.put((start, [start]))  
    visited = set()

    while not q.empty():
        node, path = q.get()
        
        if node == goal:
            return path 

        x,y = node
        for dx, dy in  [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0 and (nx,ny) not in visited:
                q.put(((nx, ny), path + [(nx, ny)]))
                visited.add((nx, ny))

    return None 
path = bfs(start, goal)
if path:
    print("Shortest Path Found (BFS):")
    print(path)
    print(f"Total Steps: {len(path) - 1}")
else:
    print(" No Path Found.")

o/p:
Shortest Path Found (BFS):
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3), (3, 4)]
Total Steps: 7
