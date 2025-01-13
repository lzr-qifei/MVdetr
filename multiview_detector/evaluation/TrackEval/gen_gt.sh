##生成MOT格式的gt用于计算HOTA，如果已有可忽略##
python /home/lizirui/MVdetr/multiview_detector/evaluation/TrackEval/gt2HOTAgt.py \
    --input /home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/wild_results/mota_pred.txt \
    --output /home/lizirui/TrackEval/MOT/pred/output.txt