# from multiview_detector.evaluation.eval_mot import mot_metrics
import motmetrics as mm
import numpy as np
import datetime
import pytz

# 获取当前时间，并设置为北京时间（UTC+8）
beijing_tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(beijing_tz)

# 获取当前时间
# current_time = datetime.datetime.now()

# 格式化时间
formatted_time = current_time.strftime("%m-%d-%H-%M")

def mot_metrics(tSource, gtSource,output_folder, scale=0.025):
    gt = np.loadtxt(gtSource, delimiter=',')
    dt = np.loadtxt(tSource, delimiter=',')

    accs = []
    frame_id = 360
    false_positives_per_frame = []
    for seq in np.unique(gt[:, 0]).astype(int):
        acc = mm.MOTAccumulator()
        for frame in np.unique(gt[:, 1]).astype(int):
            gt_dets = gt[np.logical_and(gt[:, 0] == seq, gt[:, 1] == frame)][:, (2, 8, 9)]
            dt_dets = dt[np.logical_and(dt[:, 0] == seq, dt[:, 1] == frame)][:, (2, 8, 9)]

            # format: gt, t
            C = mm.distances.norm2squared_matrix(gt_dets[:, 1:3] * scale, dt_dets[:, 1:3] * scale)
            C = np.sqrt(C)

            acc.update(gt_dets[:, 0].astype('int').tolist(),
                       dt_dets[:, 0].astype('int').tolist(),
                       C,
                       frameid=frame)
            
            events = acc.events.loc[frame_id]
            num_false_positives = events[events.Type == 'FP'].shape[0]
            false_positives_per_frame.append(num_false_positives)
            frame_id+=1
        accs.append(acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print("\n")
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    import pandas,os
    # summary.to_excel('/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/wild_results/eval_wild'+formatted_time+'.xlsx')
    ex_path = os.path.join(output_folder,f'wild-{formatted_time}.xlsx')
    summary.to_excel(ex_path)

    return

# gt = '/share2/dataset/MultiviewX/mota_gt.txt'

# pred = '/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/results/mota_pred.txt'
# mot_metrics(pred,gt,scale=0.4)