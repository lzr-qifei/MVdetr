import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
# from filterpy.kalman import KalmanFilter
import random
import matplotlib.pyplot as plt
from ocsort import OCSort
from ocsort_oppsite import OCSort_opps
import os
# from eval_wild import mot_metrics_wild
# from eval import mot_metrics
import eval_multiviewx
import eval_wild
import argparse

    
def load_detections(filename):
    """
    Loads detections from a text file.
    
    Params:
      filename - the name of the file containing the detections
    
    Returns:
      A dictionary where each key is a frame number and the value is a list of detections for that frame
    """
    detections = {}
    with open(filename, 'r') as f:
        for line in f:
            frame_id, x, y,score = map(float, line.strip().split(' '))
            frame_id = int(frame_id)
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append([x, y,score])
    return detections

# def save_tracking_results(filename, results):
#     """
#     Saves tracking results to a text file.
    
#     Params:
#       filename - the name of the file to save the results
#       results - a list of tracking results, each result is [frame_id, x, y, id]
#     """
#     with open(filename, 'w') as f:
#         for result in results:
#             f.write(','.join(map(str, result)) + '\n')
# def transform_results_to_motform(input_file,output_file):
#     def transform_line(line):
#         parts = line.strip().split(',')
#         frame_id = parts[0]
#         x = parts[1]
#         y = parts[2]
#         obj_id = parts[3]
        
#         # 创建新的格式 [0, frame_id, id, -1, -1, -1, -1, 1, x, y, -1]
#         new_line = f"0,{frame_id},{obj_id},-1,-1,-1,-1,1,{x},{y},-1"
#         return new_line

#     def transform_file(input_file, output_file):
#         with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#             for line in infile:
#                 new_line = transform_line(line)
#                 outfile.write(new_line + '\n')

#     # 转换文件
#     transform_file(input_file, output_file)
def save_and_transform_tracking_results(results, output_file):
    """
    Saves tracking results to a text file and transforms them into MOT format in a single step.
    
    Params:
      filename - the name of the file to save the results
      results - a list of tracking results, each result is [frame_id, x, y, id]
      output_file - the file to save the transformed results in MOT format
    """
    tmpfile =  os.path.join(os.path.dirname(output_file),'tmp_results.txt')
    # 1. Save the tracking results to the file
    with open(tmpfile, 'w') as f:
        for result in results:
            f.write(','.join(map(str, result)) + '\n')

    # 2. Transform the results to MOT format and save to output file
    def transform_line(line):
        parts = line.strip().split(',')
        frame_id = parts[0]
        x = parts[1]
        y = parts[2]
        obj_id = parts[3]
        
        # Create new format [0, frame_id, id, -1, -1, -1, -1, 1, x, y, -1]
        new_line = f"0,{frame_id},{obj_id},-1,-1,-1,-1,1,{x},{y},-1"
        return new_line

    def transform_file(input_file, output_file):
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                new_line = transform_line(line)
                outfile.write(new_line + '\n')

    # Call the transform function
    transform_file(tmpfile, output_file)


def visualize_tracking_results(tracking_results, save_path=None,dataset='wild'):
    """
    Visualizes the tracking results using matplotlib.
    
    Params:
      tracking_results - a list of tracking results, each result is [frame_id, x, y, id]
      save_path - optional path to save the plot (e.g., 'output.png')
    """
    # Create a color map for each ID
    id_colors = {}
    def get_color(trk_id):
        if trk_id not in id_colors:
            id_colors[trk_id] = (random.random(), random.random(), random.random())
        return id_colors[trk_id]

    # Group results by ID
    tracks = {}
    for result in tracking_results:
        frame_id, x, y, trk_id = result
        frame_id,trk_id = int(frame_id),int(trk_id)
        if trk_id not in tracks:
            tracks[trk_id] = []
        tracks[trk_id].append((frame_id, x, y))

    # Plot all frames in a single figure with higher resolution
    plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI
    plt.title('Tracking Results')
    if dataset=='wild':
        plt.xlim(0, 480)  # Adjust based on your data range
        plt.ylim(0, 1440)  # Adjust based on your data range
    else:
        plt.xlim(0, 1000)  # Adjust based on your data range
        plt.ylim(0, 640)  # Adjust based on your data range

    for trk_id, track in tracks.items():
        # Sort by frame_id to ensure the trajectory is plotted correctly
        track = sorted(track)  # Sort by frame_id
        xs = [x for _, x, _ in track]
        ys = [y for _, _, y in track]
        color = get_color(trk_id)
        
        # Draw the full trajectory
        plt.plot(xs, ys, color=color, label=f'ID {trk_id}')
        plt.scatter(xs, ys, color=color)  # Draw the points

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # plt.legend(loc=None)
    # plt.legend.set_visible(False)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # Save with higher DPI

def main():
    # 从命令行获取输入参数
    parser = argparse.ArgumentParser(description="Tracking with SORT tracker")
    parser.add_argument('--det_path', type=str, required=True, help="detection file")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save the output")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name",default='wild_40_pred')
    parser.add_argument('--eval', type=str, help="dataset to eval",default='wild')
    parser.add_argument('--gt', type=str, help="gt path",default='')

    args = parser.parse_args()

    # 从命令行参数中获取输入和输出文件夹路径以及实验名称
    det_path = args.det_path
    output_folder = args.output_folder
    exp_name = args.exp_name
    if args.eval == 'wild':
        evaluator = eval_wild.mot_metrics_wild
    elif args.eval == 'multiviewx':
        evaluator = eval_multiviewx.mot_metrics
    else:
        evaluator = None

    # 构建输出文件路径
    input_file = det_path
    output_file = os.path.join(output_folder, f'{exp_name}.txt')
    vis_path = os.path.join(output_folder, 'path.svg')
    gtfile = args.gt

    # 加载检测结果
    detections = load_detections(input_file)

    # 初始化 SORT tracker
    tracker = OCSort(det_thresh=0.1, use_byte=False)
    # tracker = OCSort_opps(det_thresh=0.1, use_byte=False)

    # 存储跟踪结果
    tracking_results = []

    # 按帧处理每一帧的检测结果
    for frame_id in sorted(detections.keys()):
        dets = np.array(detections[frame_id])
        trackers = tracker.update(dets)

        for trk in trackers:
            x, y, trk_id = trk
            trk_id = int(trk_id)
            tracking_results.append([frame_id, x, y, trk_id])

    # 保存跟踪结果
    # save_tracking_results(output_file, tracking_results)
    print('dist_mean: ',tracker.dist_mean/tracker.frame_count)
    save_and_transform_tracking_results(tracking_results, output_file)
    # 可视化跟踪结果
    visualize_tracking_results(tracking_results, vis_path,args.eval)
    if evaluator is not None:
        evaluator(output_file,gtfile,output_folder,1)

if __name__ == "__main__":
    main()
