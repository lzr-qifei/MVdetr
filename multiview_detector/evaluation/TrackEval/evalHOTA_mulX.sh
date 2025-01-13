python ./run/run_mot_challenge.py \
 --BENCHMARK MOT17 --TRACKERS_TO_EVAL output.txt \
 --GT_FOLDER  /home/lizirui/TrackEval/MOT/gt --TRACKERS_FOLDER /home/lizirui/TrackEval/MOT/pred \
 --METRICS CLEAR HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1  