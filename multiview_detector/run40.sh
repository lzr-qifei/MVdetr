srun -p ec529f6c-0264-4219-98c7-52d0c710f715 --workspace-id f7545db2-737e-46cc-a8c8-2a266efc6616 \
 -N 1 -f pt -d StandAlone -r N5IP.nn.I90.1 -j mvdetr_r34_train  \
bash -c 'export NCCL_IB_GID_INDEX=5 && export NCCL_P2P_DISABLE=1 && export OMP_NUM_THREADS=1 && source activate /home/mnt/lizirui/envs/TrackTacular \
 && python /home/mnt/lizirui/MVDeTr-main/main.py  -d multiviewx --data /home/mnt/lizirui/data/MultiviewX --world_reduce 2 --arch resnet34'

