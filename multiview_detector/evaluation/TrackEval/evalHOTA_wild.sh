python ./run/run_wildtrack.py \
 --BENCHMARK MOT17 --TRACKERS_TO_EVAL output.txt --GT_TO_EVAL wild_gt.txt \
 --GT_FOLDER  /home/lizirui/TrackEval/MOT/gt --TRACKERS_FOLDER /home/lizirui/TrackEval/MOT/pred \
 --METRICS CLEAR HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1  