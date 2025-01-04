# 读取txt文件并进行处理
def process_tracking_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    processed_lines = []
    
    # 遍历每一行，转换数据
    for line in lines:
        # 将行数据按空格分割
        parts = line.strip().split(',')
        
        # 提取所需的数据并创建新的行格式
        frame_id = parts[1]
        target_id = parts[2]
        x = parts[8]
        y = parts[9]
        
        new_line = f"{frame_id},{target_id},{x},{y},0,0,1,1,1.000000\n"
        processed_lines.append(new_line)
    
    # 按照target_id升序排序
    processed_lines.sort(key=lambda x: int(x.split(',')[1]))  # 按 target_id 升序排序
    
    # 将处理后的数据写入输出文件
    with open(output_file, 'w') as outfile:
        outfile.writelines(processed_lines)

# # 调用函数处理文件
# input_file = '/home/lizirui/det_results/earlybird/mota_pred_wildtrack.txt'  # 输入文件路径
# output_file = '/home/lizirui/TrackEval/MOT/pred/earlybird.txt'  # 输出文件路径
# process_tracking_file(input_file, output_file)
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process tracking files.')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided file paths
    process_tracking_file(args.input, args.output)
