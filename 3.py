# Simple DFS Traversal (Game Map)

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['G'],
    'F': [],
    'G': []
}

def dfs(start, goal, graph):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node == goal:
            print("✅ Goal Found:", path)
            return
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(graph[node]):
                stack.append((neighbor, path + [neighbor]))
    print("❌ Goal Not Found")

# Run DFS
dfs('A', 'G', graph)
