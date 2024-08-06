def split_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 用于存储每个文件的内容
    split_data = {}

    # 遍历每一行数据
    for line in lines:
        # 解析数据
        frame_id, x, y, id_ = map(int, line.strip().strip('[]').split(' '))

        # 计算当前frame_id所属的文件
        file_index = frame_id // 10

        if file_index not in split_data:
            split_data[file_index] = []

        split_data[file_index].append(line)

    # 写入拆分后的文件
    for index, data in split_data.items():
        output_file = f"/root/autodl-tmp/MultiviewX/gts/seq_{index}.txt"
        with open(output_file, 'w') as file:
            file.writelines(data)

# 使用示例
split_file('/root/autodl-tmp/MultiviewX/gt_id.txt')
