srun -p c7e6fad6-4dfa-42ef-af06-b0858c594d44 --workspace-id c75ef8a9-625f-4985-8c14-67b04e72a8c1 \
 -N 1 -f pt -d StandAlone -r N1lS.Ia.I20.1  -j mvdetr_r50_train  \
bash -c 'export OMP_NUM_THREADS=1  && source activate /home/mnt/lizirui/envs/TrackTacular \
 && python /home/mnt/lizirui/MVDeTr-main/main.py  -d multiviewx --data /home/mnt/lizirui/data/MultiviewX --world_reduce 4 --arch resnet50 \
 --pth /home/mnt/lizirui/MVDeTr-main/multiview_detector/r50.pth --out_path /home/mnt/lizirui/MVDeTr-main/multiview_detector/results/test.txt \
 --det_thres 0.65 --epochs 150 --sensecore'

