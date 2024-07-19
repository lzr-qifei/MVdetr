import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from filterpy.kalman import KalmanFilter
import random
import matplotlib.pyplot as plt
class KalmanPointTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as a point [x, y].
    """
    count = 0

    def __init__(self, point):
        """
        Initializes a tracker using the initial point [x, y].
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.kf.R *= 10.
        self.kf.P[2:, 2:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[2:, 2:] *= 0.01

        self.kf.x[:2] = convert_point_to_z(point)
        self.time_since_update = 0
        self.id = KalmanPointTracker.count
        KalmanPointTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, point):
        """
        Updates the state vector with observed point.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_point_to_z(point))

    def predict(self):
        """
        Advances the state vector and returns the predicted point estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_point(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current point estimate.
        """
        return convert_x_to_point(self.kf.x)

def convert_point_to_z(point):
    """
    Converts a point [x, y] to the state format [x, y].
    """
    return np.array(point).reshape((2, 1))

def convert_x_to_point(x):
    """
    Converts the Kalman filter state to point format [x, y].
    """
    return np.array([x[0], x[1]]).flatten()

def associate_detections_to_trackers(detections, trackers, distance_threshold=30):
    """
    Assigns detections to tracked objects (both represented as points [x, y]).

    Returns 3 lists: matches, unmatched_detections, and unmatched_trackers.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int)

    distance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            distance_matrix[d, t] = np.linalg.norm(det - trk)

    if min(distance_matrix.shape) > 0:
        matched_indices = linear_assignment(distance_matrix)
        matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with high distance
    matches = []
    for m in matched_indices:
        if distance_matrix[m[0], m[1]] > distance_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=10, min_hits=1, distance_threshold=40):
        """
        Sets key parameters for SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 2))):
        """
        Params:
          dets - a numpy array of detections in the format [[x, y], [x, y], ...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 2)) for frames without detections).
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.distance_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanPointTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 3))
    
import numpy as np



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
            frame_id, x, y = map(float, line.strip().split(' '))
            frame_id = int(frame_id)
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append([x, y])
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
def visualize_tracking_results(tracking_results):
    """
    Visualizes the tracking results using matplotlib.
    
    Params:
      tracking_results - a list of tracking results, each result is [frame_id, x, y, id]
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
        if trk_id not in tracks:
            tracks[trk_id] = []
        tracks[trk_id].append((frame_id, x, y))

    # Plot all frames in a single figure
    plt.figure()
    # plt.title('Tracking Results')
    plt.title('GT Results')
    plt.xlim(0, 1000)  # Adjust based on your data range
    plt.ylim(0, 640)  # Adjust based on your data range

    for trk_id, track in tracks.items():
        track = sorted(track)  # Sort by frame_id
        xs = [x for _, x, _ in track]
        ys = [y for _, _, y in track]
        color = get_color(trk_id)
        plt.plot(xs, ys, color=color, label=f'ID {trk_id}')
        plt.scatter(xs, ys, color=color)

    plt.legend(loc='upper right')
    plt.show()

def main():
    input_file = '/home/SENSETIME/lizirui/utils/pts_tracker/pred_40.txt'
    output_file = '/home/SENSETIME/lizirui/utils/pts_tracker/pred_40_track_result.txt'

    # Load detections from file
    detections = load_detections(input_file)

    # Initialize SORT tracker
    tracker = Sort()

    # Store the tracking results
    tracking_results = []
    gt_results = []
    # Process each frame in order
    for frame_id in sorted(detections.keys()):
        dets = np.array(detections[frame_id])
        trackers = tracker.update(dets)

        for trk in trackers:
            x, y, trk_id = trk
            trk_id = int(trk_id)
            tracking_results.append([frame_id, x, y, trk_id])
    with open('/home/SENSETIME/lizirui/utils/pts_tracker/gt_40.txt', 'r') as f:
        for line in f:
            frame_id, gt_x, gt_y,id = map(int, line.strip().split(' '))
            gt_results.append([frame_id, gt_x, gt_y,id])
    # Save tracking results to file
    save_tracking_results(output_file, tracking_results)
    # visualize_tracking_results(tracking_results)
    visualize_tracking_results(gt_results)

if __name__ == "__main__":
    main()

