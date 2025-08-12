from collections import deque, defaultdict

#given image
graph = defaultdict(list)
graph[1] = [2, 7, 8]
graph[2] = [3, 6]
graph[3] = [4, 5]
graph[8] = [9, 12]
graph[9] = [10]
graph[12] = [11]

# DFS implementation (recursive)
def dfs(node, visited):
    if node not in visited:
        print(node, end=' ')
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor, visited)

# BFS implementation (using queue)
def bfs(start):
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=' ')
            visited.add(node)
            for neighbor in graph[node]:
                queue.append(neighbor)

# Run DFS and BFS from node 1
print("DFS Traversal:")
dfs(1, set())

print("\nBFS Traversal:")
bfs(1)
