
##将Track结果转换为计算mot指标要求的格式，直接输出到output_file
##输入的txt每行格式应该是[frame_id,x,y,id]
def transform_results_to_motform(input_file,output_file):
    def transform_line(line):
        parts = line.strip().split(',')
        frame_id = parts[0]
        x = parts[1]
        y = parts[2]
        obj_id = parts[3]
        
        # 创建新的格式 [0, frame_id, id, -1, -1, -1, -1, 1, x, y, -1]
        new_line = f"0,{frame_id},{obj_id},-1,-1,-1,-1,1,{x},{y},-1"
        return new_line

    def transform_file(input_file, output_file):
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                new_line = transform_line(line)
                outfile.write(new_line + '\n')

    # # 输入文件和输出文件路径
    # input_file = '/home/SENSETIME/lizirui/utils/pts_tracker/pred_40_track_result.txt'
    # output_file = '/home/SENSETIME/lizirui/utils/pts_tracker/pred_40_track_result_mot.txt'

    # 转换文件
    transform_file(input_file, output_file)

# input_file = '/root/BEV_OC/WildTrack/pred_40_track_result.txt'
# output_file = '/root/BEV_OC/WildTrack/mota_pred.txt'
input_file = '/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/results/pred_40_track_result.txt'
output_file = '/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/results/mota_pred.txt'
# input_file = '/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/wild_results/pred_40_track_result.txt'
# output_file = '/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/wild_results/mota_pred.txt'

transform_results_to_motform(input_file,output_file)
