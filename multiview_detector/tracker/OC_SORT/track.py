import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
# from filterpy.kalman import KalmanFilter
import random
import matplotlib.pyplot as plt
from ocsort import OCSort

    
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

def save_tracking_results(filename, results):
    """
    Saves tracking results to a text file.
    
    Params:
      filename - the name of the file to save the results
      results - a list of tracking results, each result is [frame_id, x, y, id]
    """
    with open(filename, 'w') as f:
        for result in results:
            f.write(','.join(map(str, result)) + '\n')
import random
import matplotlib.pyplot as plt


def visualize_tracking_results(tracking_results, save_path=None):
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
    plt.figure(figsize=(12, 8), dpi=100)  # Set figure size and DPI
    plt.title('GT Results')
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
    plt.legend(loc='upper right')

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # Save with higher DPI

def main():
    input_file = '/root/test(1).txt'
    output_file = '/root/MVdetr/multiview_detector/tracker/OC_SORT/results/pred_40_track_result.txt'
    vis_path = '/root/MVdetr/multiview_detector/tracker/OC_SORT/results/1.png'

    # Load detections from file
    detections = load_detections(input_file)

    # Initialize SORT tracker
    tracker = OCSort(det_thresh=0.3,use_byte=True)

    # Store the tracking results
    tracking_results = []
    gt_results = []
    pred_results = []
    # Process each frame in order
    for frame_id in sorted(detections.keys()):
        dets = np.array(detections[frame_id])
        trackers = tracker.update(dets)

        for trk in trackers:
            x, y, trk_id = trk
            trk_id = int(trk_id)
            tracking_results.append([frame_id, x, y, trk_id])
    with open('/root/MVdetr/multiview_detector/tracker/OC_SORT/test.txt', 'r') as f:
        for line in f:
            frame_id, gt_x, gt_y,id = map(float, line.strip().split(','))
            pred_results.append([frame_id, gt_x, gt_y,id])
    # Save tracking results to file
    save_tracking_results(output_file, tracking_results)
    # visualize_tracking_results(tracking_results)
    visualize_tracking_results(tracking_results,vis_path)

if __name__ == "__main__":
    main()

