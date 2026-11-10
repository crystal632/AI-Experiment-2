"""
实验二：题目一（迷宫寻路）与题目二（加权地形）

接口：
1. astar_maze(start, goal, grid, use_heuristic=True, stats=None)
   - grid: 'S'、'.'、'X'、'E' 组成的二维列表
   - 返回：最短路径坐标列表；无解返回 []

2. astar_weighted(start, goal, grid, use_heuristic=True, stats=None)
   - grid: 数字地形，起点可写 'S'，终点可写 'E'
   - 进入普通数字格的代价为该格数字；进入 E 的代价为 0
   - 返回：最小总代价；无解返回 -1

两个函数均只允许上下左右移动。

stats（可选）：
   - 传入空字典 {}，函数会写入 expanded_nodes。
   - expanded_nodes 统计“从优先队列取出、且不是过期记录”的节点数。
"""

import heapq
from itertools import count
from math import inf

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def manhattan(a, b):
    """曼哈顿距离。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _validate_grid(grid):
    if not grid or not grid[0]:
        raise ValueError("地图不能为空")
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图每一行的长度必须一致")
    return len(grid), cols


def _validate_point(point, rows, cols, name):
    if (not isinstance(point, tuple) or len(point) != 2
            or not all(isinstance(v, int) for v in point)):
        raise ValueError(f"{name}必须是 (row, col) 整数坐标")
    r, c = point
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError(f"{name}超出地图范围")


def astar_maze(start, goal, grid, use_heuristic=True, stats=None):
    """
    题目一：普通 A* 迷宫寻路。
    返回从 start 到 goal 的坐标序列；无解返回 []。
    use_heuristic=False 时 h(n)=0。
    stats 可选：记录有效出队节点数 expanded_nodes。
    """
    rows, cols = _validate_grid(grid)
    _validate_point(start, rows, cols, "起点")
    _validate_point(goal, rows, cols, "终点")

    if stats is not None:
        stats["expanded_nodes"] = 0

    def blocked(point):
        return grid[point[0]][point[1]] == "X"

    if blocked(start) or blocked(goal):
        return []

    sequence = count()
    queue = []
    start_h = manhattan(start, goal) if use_heuristic else 0
    heapq.heappush(queue, (start_h, next(sequence), 0, start))

    g_cost = {start: 0}
    parent = {start: None}

    while queue:
        _, _, current_cost, current = heapq.heappop(queue)

        # 丢弃优先队列中已经过期的旧记录
        if current_cost != g_cost.get(current):
            continue

        # 统计有效出队节点（包括最终取出的目标节点）
        if stats is not None:
            stats["expanded_nodes"] += 1

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]

        r, c = current
        for dr, dc in DIRECTIONS:
            neighbor = (r + dr, c + dc)
            nr, nc = neighbor

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if blocked(neighbor):
                continue

            new_cost = current_cost + 1
            if new_cost < g_cost.get(neighbor, inf):
                g_cost[neighbor] = new_cost
                parent[neighbor] = current
                h = manhattan(neighbor, goal) if use_heuristic else 0
                heapq.heappush(
                    queue, (new_cost + h, next(sequence), new_cost, neighbor)
                )

    return []


def astar_weighted(start, goal, grid, use_heuristic=True, stats=None):
    """
    题目二：加权地形 A*。
    数字格的进入代价等于格子权值；进入终点 E 的代价为 0。
    起点格不计入路径代价。无解返回 -1。

    启发函数：
    终点进入代价为 0，因此曼哈顿距离中的最后一步不能计入
    最小普通地形代价。使用 max(曼哈顿距离 - 1, 0) * 最小普通格代价。
    """
    rows, cols = _validate_grid(grid)
    _validate_point(start, rows, cols, "起点")
    _validate_point(goal, rows, cols, "终点")

    if stats is not None:
        stats["expanded_nodes"] = 0

    def is_blocked(value):
        return value == "X"

    if is_blocked(grid[start[0]][start[1]]) or is_blocked(grid[goal[0]][goal[1]]):
        return -1

    traversable_costs = [
        value
        for row in grid
        for value in row
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    ]
    min_step_cost = min(traversable_costs, default=0)

    def entry_cost(point):
        value = grid[point[0]][point[1]]
        if point == goal or value == "E":
            return 0
        if value == "S":
            return 0
        if (isinstance(value, (int, float))
                and not isinstance(value, bool) and value >= 0):
            return value
        raise ValueError(f"无法识别的地形值：{value!r}")

    def heuristic(point):
        if not use_heuristic:
            return 0
        # 最后一步进入 E 的代价为 0，故只估算此前至少要进入的格子
        remaining_paid_steps = max(manhattan(point, goal) - 1, 0)
        return remaining_paid_steps * min_step_cost

    sequence = count()
    queue = []
    heapq.heappush(queue, (heuristic(start), next(sequence), 0, start))

    g_cost = {start: 0}
    while queue:
        _, _, current_cost, current = heapq.heappop(queue)

        if current_cost != g_cost.get(current):
            continue

        if stats is not None:
            stats["expanded_nodes"] += 1

        if current == goal:
            return current_cost

        r, c = current
        for dr, dc in DIRECTIONS:
            neighbor = (r + dr, c + dc)
            nr, nc = neighbor

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            value = grid[nr][nc]
            if is_blocked(value):
                continue

            step_cost = entry_cost(neighbor)
            new_cost = current_cost + step_cost
            if new_cost < g_cost.get(neighbor, inf):
                g_cost[neighbor] = new_cost
                heapq.heappush(
                    queue,
                    (new_cost + heuristic(neighbor),
                     next(sequence), new_cost, neighbor)
                )

    return -1


def _run_teacher_examples():
    """运行实验指导书中的题目一、题目二测试用例。"""
    maze = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    start1, goal1 = (0, 0), (3, 4)

    maze_stats = {}
    path = astar_maze(start1, goal1, maze, stats=maze_stats)
    zero_stats = {}
    zero_path = astar_maze(
        start1, goal1, maze, use_heuristic=False, stats=zero_stats
    )

    print("题目一：迷宫寻路")
    print("最短路径：", path)
    print("路径长度（移动步数）：", len(path) - 1 if path else -1)
    print("曼哈顿启发函数搜索节点数：", maze_stats["expanded_nodes"])
    print("h=0 路径：", zero_path)
    print("h=0 搜索节点数：", zero_stats["expanded_nodes"])

    weighted_map = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    start2, goal2 = (0, 0), (4, 4)

    weighted_stats = {}
    cost = astar_weighted(start2, goal2, weighted_map, stats=weighted_stats)
    weighted_zero_stats = {}
    zero_cost = astar_weighted(
        start2, goal2, weighted_map,
        use_heuristic=False, stats=weighted_zero_stats
    )

    print("\n题目二：加权地形")
    print("最小总代价：", cost)
    print("修正启发函数搜索节点数：", weighted_stats["expanded_nodes"])
    print("h=0 对照结果：", zero_cost)
    print("h=0 搜索节点数：", weighted_zero_stats["expanded_nodes"])


if __name__ == "__main__":
    _run_teacher_examples()
