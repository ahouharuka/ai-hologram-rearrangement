import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def calculate_distance_squared(pos1, pos2):
    """计算两点之间距离的平方（避免开方运算，提高效率）"""
    return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

def hungarian_atom_rearrangement(initial_positions, target_positions):
    """
    使用匈牙利算法计算原子重排的最佳路径
    
    参数:
    initial_positions: 初始原子位置的列表，格式为 [(x1, y1), (x2, y2), ...]
    target_positions: 目标原子位置的列表，格式为 [(x1, y1), (x2, y2), ...]
    
    返回:
    assignment: 分配结果，表示每个初始原子应该移动到哪个目标位置
    total_cost: 总移动成本（距离平方和）
    """
    # 确保原子数量匹配
    n_atoms = len(initial_positions)
    n_targets = len(target_positions)
    
    if n_atoms > n_targets:
        raise ValueError("原子数量不能超过目标位置数量")
    
    # 构建成本矩阵（距离平方）
    cost_matrix = np.zeros((n_atoms, n_targets))
    for i in range(n_atoms):
        for j in range(n_targets):
            cost_matrix[i, j] = calculate_distance_squared(initial_positions[i], target_positions[j])
    
    # 使用匈牙利算法求解最优分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 计算总成本
    total_cost = cost_matrix[row_ind, col_ind].sum()
    
    # 构建分配结果
    assignment = list(zip(row_ind, col_ind))
    
    return assignment, total_cost, cost_matrix

def visualize_rearrangement(initial_positions, target_positions, assignment):
    """可视化原子重排过程"""
    plt.figure(figsize=(10, 8))
    
    # 绘制初始位置
    initial_x = [pos[0] for pos in initial_positions]
    initial_y = [pos[1] for pos in initial_positions]
    plt.scatter(initial_x, initial_y, c='blue', s=100, label='初始位置', alpha=0.7)
    
    # 绘制目标位置
    target_x = [pos[0] for pos in target_positions]
    target_y = [pos[1] for pos in target_positions]
    plt.scatter(target_x, target_y, c='red', s=100, marker='x', label='目标位置', alpha=0.7)
    
    # 绘制移动路径
    for i, j in assignment:
        plt.plot([initial_positions[i][0], target_positions[j][0]], 
                 [initial_positions[i][1], target_positions[j][1]], 
                 'g--', alpha=0.5)
        # 添加箭头表示方向
        plt.arrow(initial_positions[i][0], initial_positions[i][1],
                  (target_positions[j][0] - initial_positions[i][0]) * 0.8,
                  (target_positions[j][1] - initial_positions[i][1]) * 0.8,
                  head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.5)
    
    plt.title('原子阵列重排路径规划')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 生成随机初始位置和目标位置
    np.random.seed(42)  # 设置随机种子以确保可重复性
    
    n_atoms = 10
    initial_positions = np.random.rand(n_atoms, 2) * 10  # 在10x10区域内随机生成初始位置
    
    # 创建规则的目标阵列（例如3x4网格，但可能有多余位置）
    target_grid_size = (3, 4)
    target_positions = []
    for i in range(target_grid_size[0]):
        for j in range(target_grid_size[1]):
            target_positions.append((2 + j * 2, 2 + i * 2))  # 2的间距
    
    # 确保目标位置数量足够
    if len(target_positions) < n_atoms:
        # 添加额外目标位置
        extra_positions = np.random.rand(n_atoms - len(target_positions), 2) * 10
        target_positions.extend([(pos[0], pos[1]) for pos in extra_positions])
    
    # 计算最佳分配
    assignment, total_cost, cost_matrix = hungarian_atom_rearrangement(initial_positions, target_positions)
    
    print("分配结果 (初始原子索引 -> 目标位置索引):")
    for i, j in assignment:
        print(f"原子 {i} -> 目标 {j}")
    
    print(f"\n总移动成本 (距离平方和): {total_cost:.2f}")
    
    # 可视化结果
    visualize_rearrangement(initial_positions, target_positions, assignment)
    
    # 打印成本矩阵（可选）
    print("\n成本矩阵 (距离平方):")
    print(cost_matrix)