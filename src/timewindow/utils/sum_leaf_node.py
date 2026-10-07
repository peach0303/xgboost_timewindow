import json

def sum_leave_nodes(json_data_raw):
    
    def count_leaves(node):
        if 'leaf' in node:
            return 1
        else:
            left_count = count_leaves(node['children'][0])
            right_count = count_leaves(node['children'][1])
            return left_count + right_count

    json_data = []
    for x in json_data_raw:
        json_data.append(json.loads(x))


    all_leaves_count = 0
    for root_node in json_data:
        # 计算叶子节点数
        all_leaves_count += count_leaves(root_node)
    print("叶子节点数:", all_leaves_count)
    return all_leaves_count
