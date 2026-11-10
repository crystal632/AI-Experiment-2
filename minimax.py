
"""
实验二：题目三——极大极小博弈树的最佳得分

根节点为 MAX 玩家。
实现普通 Minimax 和 Alpha-Beta 剪枝算法。
支持从左到右、从右到左访问子节点。
统计访问节点数和剪枝子节点数。
"""


class TreeNode:
    """博弈树节点"""

    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


def minimax(node, is_maximizing=True, stats=None):
    """
    普通 Minimax 算法。

    stats：可选统计字典，记录访问节点数。
    """
    if stats is not None:
        stats["visited_nodes"] = stats.get("visited_nodes", 0) + 1

    if not node.children:
        if node.value is None:
            raise ValueError("叶子节点必须提供 value")
        return node.value

    child_values = [
        minimax(child, not is_maximizing, stats)
        for child in node.children
    ]

    if is_maximizing:
        return max(child_values)
    else:
        return min(child_values)


def alpha_beta(
    node,
    alpha=float("-inf"),
    beta=float("inf"),
    is_maximizing=True,
    stats=None,
    order="left"
):
    """
    Alpha-Beta 剪枝算法。

    order="left"：从左到右访问
    order="right"：从右到左访问

    stats：
    visited_nodes：实际访问节点数
    pruned_children：被跳过的直接子节点数
    """
    if stats is None:
        stats = {
            "visited_nodes": 0,
            "pruned_children": 0
        }

    stats["visited_nodes"] = stats.get("visited_nodes", 0) + 1
    stats["pruned_children"] = stats.get("pruned_children", 0)

    if order not in ("left", "right"):
        raise ValueError("order 必须是 'left' 或 'right'")

    if not node.children:
        if node.value is None:
            raise ValueError("叶子节点必须提供 value")
        return node.value

    children = (
        node.children
        if order == "left"
        else list(reversed(node.children))
    )

    if is_maximizing:
        best = float("-inf")

        for index, child in enumerate(children):
            value = alpha_beta(
                child, alpha, beta, False, stats, order
            )

            best = max(best, value)
            alpha = max(alpha, best)

            if alpha >= beta:
                stats["pruned_children"] += (
                    len(children) - index - 1
                )
                break

        return best

    else:
        best = float("inf")

        for index, child in enumerate(children):
            value = alpha_beta(
                child, alpha, beta, True, stats, order
            )

            best = min(best, value)
            beta = min(beta, best)

            if alpha >= beta:
                stats["pruned_children"] += (
                    len(children) - index - 1
                )
                break

        return best


def build_teacher_tree():
    """根据原有题目三测试树构建博弈树"""

    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])

    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])

    root = TreeNode(children=[min1, min2])

    return root


def main():

    print("=" * 55)
    print("实验二：题目三——Minimax 与 Alpha-Beta")
    print("=" * 55)

    # 普通 Minimax
    root = build_teacher_tree()
    minimax_stats = {"visited_nodes": 0}

    minimax_result = minimax(
        root,
        stats=minimax_stats
    )

    print("\n普通 Minimax 算法：")
    print("最佳得分：", minimax_result)
    print("访问节点数：", minimax_stats["visited_nodes"])

    # Alpha-Beta 从左到右
    root = build_teacher_tree()
    left_stats = {
        "visited_nodes": 0,
        "pruned_children": 0
    }

    left_result = alpha_beta(
        root,
        stats=left_stats,
        order="left"
    )

    print("\nAlpha-Beta（从左到右）：")
    print("最佳得分：", left_result)
    print("访问节点数：", left_stats["visited_nodes"])
    print("剪枝子节点数：", left_stats["pruned_children"])

    # Alpha-Beta 从右到左
    root = build_teacher_tree()
    right_stats = {
        "visited_nodes": 0,
        "pruned_children": 0
    }

    right_result = alpha_beta(
        root,
        stats=right_stats,
        order="right"
    )

    print("\nAlpha-Beta（从右到左）：")
    print("最佳得分：", right_result)
    print("访问节点数：", right_stats["visited_nodes"])
    print("剪枝子节点数：", right_stats["pruned_children"])

    # 验证
    assert minimax_result == 5
    assert left_result == 5
    assert right_result == 5

    print("\n测试通过！")
    print("老师预期结果：5")
    print("Minimax 实际结果：", minimax_result)
    print("Alpha-Beta 左序结果：", left_result)
    print("Alpha-Beta 右序结果：", right_result)


if __name__ == "__main__":
    main()
