"""
实验二：题目四——竞争环境下的宝藏分配

严格按实验指导书第14—15页实现：
- A 为 MAX，B 为 MIN，行动顺序 A -> B -> A -> B。
- X 为障碍；数字为进入该格的代价；A、B、T 格进入代价均为 1。
- 只能上下左右移动。
- 经过其他宝藏不自动收集；已收集宝藏格仍可通行。
- 双方不互相阻挡。
- Gain(p, Ti) = Vi - d(p, Ti)，U = Score_A - Score_B。
- H=2 预期：T2，-1；H=4 预期：T3，3。
"""

import heapq
from itertools import count
from math import inf

# 地图中 A、B、T 格统一按进入代价 1 处理。
GRID = [
    [1, 1, 1, "X", 2, 1, 1],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, 1, 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, 1, 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    [1, 1, 1, "X", 1, 1, 1],
]

LOCATIONS = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}

TREASURE_VALUES = {
    "T1": 9,
    "T2": 6,
    "T3": 7,
    "T4": 7,
}

TREASURE_ORDER = ("T1", "T2", "T3", "T4")
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def manhattan(a, b):
    """曼哈顿距离；本地图所有可进入格的代价至少为 1，因此该启发函数可采纳。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal):
    """
    A* 计算从 start 到 goal 的最小移动代价。
    代价按“进入相邻格”的格子权值累计，起点本身不计费。
    """
    rows, cols = len(GRID), len(GRID[0])
    sequence = count()

    # 所有可通行格的最小进入代价为 1
    min_step_cost = min(
        cell for row in GRID for cell in row
        if isinstance(cell, (int, float)) and cell > 0
    )

    open_heap = [
        (manhattan(start, goal) * min_step_cost, next(sequence), 0, start)
    ]
    g_cost = {start: 0}

    while open_heap:
        _, _, current_g, current = heapq.heappop(open_heap)

        if current_g != g_cost.get(current):
            continue

        if current == goal:
            return current_g

        row, col = current
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            step_cost = GRID[nr][nc]
            if step_cost == "X":
                continue

            neighbor = (nr, nc)
            new_g = current_g + step_cost

            if new_g < g_cost.get(neighbor, inf):
                g_cost[neighbor] = new_g
                h = manhattan(neighbor, goal) * min_step_cost
                heapq.heappush(
                    open_heap,
                    (new_g + h, next(sequence), new_g, neighbor)
                )

    return inf


def build_distance_matrix():
    """计算 A、B、T1、T2、T3、T4 之间的 6×6 A* 最小代价矩阵。"""
    names = ("A", "B", "T1", "T2", "T3", "T4")
    matrix = {}

    for source in names:
        matrix[source] = {}
        for target in names:
            matrix[source][target] = astar(
                LOCATIONS[source], LOCATIONS[target]
            )

    return names, matrix


NAMES, DISTANCE = build_distance_matrix()


def print_distance_matrix():
    print("A* 最小移动代价距离矩阵")
    print("      " + "".join(f"{name:>7}" for name in NAMES))
    for source in NAMES:
        values = "".join(
            f"{DISTANCE[source][target]:>7}"
            for target in NAMES
        )
        print(f"{source:>5} {values}")


def minimax_search(depth_limit, alpha_beta_enabled=False, order="left"):
    """
    博弈状态包含：
    双方当前位置、剩余宝藏、双方累计得分、当前行动玩家、搜索深度。

    order:
      left      按 T1、T2、T3、T4 从左到右
      right     按 T4、T3、T2、T1 从右到左
      heuristic 按当前行动者的即时净收益排序（用于比较剪枝效果）
    """
    if order not in ("left", "right", "heuristic"):
        raise ValueError("order 必须是 left、right 或 heuristic")

    stats = {
        "visited_nodes": 0,
        "pruned_children": 0,
    }

    def search(a_pos, b_pos, remaining, score_a, score_b,
               player, depth, alpha, beta, first_target):
        stats["visited_nodes"] += 1
        utility = score_a - score_b

        # 到达搜索深度或宝藏取完，返回当前收益差
        if depth >= depth_limit or not remaining:
            return utility, first_target

        current_pos = a_pos if player == "A" else b_pos
        current_name = _location_name(current_pos)
        candidates = [t for t in TREASURE_ORDER if t in remaining]

        if order == "right":
            candidates.reverse()
        elif order == "heuristic":
            # 即时净收益 = 宝藏价值 - 当前玩家到宝藏的最小代价
            candidates.sort(
                key=lambda t: (
                    TREASURE_VALUES[t] - DISTANCE[current_name][t]
                ),
                reverse=True
            )

        if player == "A":
            best_value = -inf
            best_first = first_target

            for index, target in enumerate(candidates):
                gain = TREASURE_VALUES[target] - DISTANCE[
                    _location_name(a_pos)
                ][target]

                next_first = target if depth == 0 else first_target

                value, returned_first = search(
                    target,
                    b_pos,
                    remaining - {target},
                    score_a + gain,
                    score_b,
                    "B",
                    depth + 1,
                    alpha,
                    beta,
                    next_first,
                )

                if value > best_value:
                    best_value = value
                    best_first = returned_first

                if alpha_beta_enabled:
                    alpha = max(alpha, best_value)
                    if alpha >= beta:
                        stats["pruned_children"] += len(candidates) - index - 1
                        break

            return best_value, best_first

        best_value = inf
        best_first = first_target

        for index, target in enumerate(candidates):
            gain = TREASURE_VALUES[target] - DISTANCE[
                _location_name(b_pos)
            ][target]

            value, returned_first = search(
                a_pos,
                target,
                remaining - {target},
                score_a,
                score_b + gain,
                "A",
                depth + 1,
                alpha,
                beta,
                first_target,
            )

            if value < best_value:
                best_value = value
                best_first = returned_first

            if alpha_beta_enabled:
                beta = min(beta, best_value)
                if alpha >= beta:
                    stats["pruned_children"] += len(candidates) - index - 1
                    break

        return best_value, best_first

    result, best_target = search(
        "A",
        "B",
        frozenset(TREASURE_ORDER),
        0,
        0,
        "A",
        0,
        -inf,
        inf,
        None,
    )

    return {
        "best_target": best_target,
        "best_value": result,
        "visited_nodes": stats["visited_nodes"],
        "pruned_children": stats["pruned_children"],
    }


def _location_name(position):
    """当前位置在博弈状态中以 A、B 或宝藏名称表示。"""
    if position in LOCATIONS:
        return position
    for name, coord in LOCATIONS.items():
        if coord == position:
            return name
    raise ValueError(f"未知位置：{position}")


def main():
    print("=" * 64)
    print("实验二：题目四——竞争环境下的宝藏分配")
    print("=" * 64)

    print_distance_matrix()

    for depth in (2, 4):
        print(f"\n搜索深度 H={depth}")

        plain = minimax_search(
            depth,
            alpha_beta_enabled=False,
            order="left"
        )
        print("Minimax：", plain)

        for order in ("left", "right", "heuristic"):
            pruned = minimax_search(
                depth,
                alpha_beta_enabled=True,
                order=order
            )
            print(f"Alpha-Beta（{order}）：", pruned)

        expected = {
            2: {"best_target": "T2", "best_value": -1},
            4: {"best_target": "T3", "best_value": 3},
        }[depth]

        assert plain["best_target"] == expected["best_target"]
        assert plain["best_value"] == expected["best_value"]

        for order in ("left", "right", "heuristic"):
            pruned = minimax_search(
                depth,
                alpha_beta_enabled=True,
                order=order
            )
            assert pruned["best_target"] == expected["best_target"]
            assert pruned["best_value"] == expected["best_value"]

        print("测试通过，预期结果：", expected)

    print("\n全部测试通过！")


if __name__ == "__main__":
    main()
