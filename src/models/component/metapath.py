import torch

graph = torch.load("data/MINDsmall/train/hetero_graph.pt")


def extract_top_k_meta_paths_with_dynamic_edges(graph, start_node, meta_paths, top_k=5, weight_threshold=0.0):
    """
    从异构图中提取每种元路径的前 top_k 条权值最大的路径。

    Args:
        graph (HeteroData): 异构图对象。
        start_node (int): 起始节点的索引。
        meta_paths (list[list[str]]): 元路径列表，每个元路径由节点类型组成。
        top_k (int): 每种元路径提取的最大路径数量。
        weight_threshold (float): 边权值过滤阈值。

    Returns:
        results (dict): 每种元路径提取的路径和权值，格式为 {meta_path: [(path, total_weight)]}.
    """
    results = {tuple(path): [] for path in meta_paths}

    for meta_path in meta_paths:
        def dfs(current_node, current_path, current_meta_path, current_weight, depth):
            # 当前元路径到达尾部时，记录路径和总权值
            if depth == len(meta_path) - 1:
                results[tuple(current_meta_path)].append((current_path[:], current_weight))
                return

            # 当前边类型
            src_type = meta_path[depth]
            dst_type = meta_path[depth + 1]
            relation = f'{src_type}_to_{dst_type}'
            current_edge_type = (src_type, dst_type)

            # 检查边是否存在
            if f'{relation}_edge_index' not in graph[current_edge_type]:
                return  # 当前边类型不存在

            # 获取动态命名的 edge_index 和 edge_weight
            edge_index = graph[current_edge_type][f'{relation}_edge_index']
            edge_weight = graph[current_edge_type].get(f'{relation}_edge_weight', torch.ones(edge_index.size(1)))

            # 查找从当前节点出发的边
            src, dst = edge_index
            mask = (src == current_node) & (edge_weight > weight_threshold)
            connected_nodes = dst[mask]
            connected_weights = edge_weight[mask]

            # 对边按权值排序，并仅保留前 top_k
            if len(connected_weights) > 0:
                sorted_indices = torch.argsort(connected_weights, descending=True)[:top_k]
                connected_nodes = connected_nodes[sorted_indices]
                connected_weights = connected_weights[sorted_indices]

            # 对每个下一跳递归
            for next_node, edge_w in zip(connected_nodes, connected_weights):
                current_path.append((current_node, next_node, current_edge_type))  # 添加当前边
                dfs(next_node, current_path, current_meta_path, current_weight + edge_w.item(), depth + 1)
                current_path.pop()  # 回溯

        # 启动 DFS
        dfs(start_node, [], meta_path, 0, 0)

    # 对结果排序并选取 top_k
    for path_type in results:
        results[path_type] = sorted(results[path_type], key=lambda x: x[1], reverse=True)[:top_k]

    return results


# 示例调用
meta_paths = [
    ['argument', 'topic', 'argument'],
    ['argument', 'subtopic', 'argument'],
    ['argument', 'trigger', 'argument'],
    ['argument', 'argument', 'argument'],
    ['argument', 'argument']
]

start_node = 2  # 起始 argument 节点索引
top_k = 5  # 每种元路径提取的最大路径数量
extracted_top_paths = extract_top_k_meta_paths_with_dynamic_edges(graph, start_node, meta_paths, top_k=top_k,
                                                                  weight_threshold=0.5)

# 打印提取结果
for path_type, paths in extracted_top_paths.items():
    print(f"Meta-path: {path_type}")
    for path, total_weight in paths:
        print(f"  Path: {path}, Total Weight: {total_weight:.2f}")