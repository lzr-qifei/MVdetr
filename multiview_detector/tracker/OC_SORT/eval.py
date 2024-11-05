# from multiview_detector.evaluation.eval_mot import mot_metrics
import motmetrics as mm
import numpy as np


def mot_metrics(tSource, gtSource, scale=0.025):
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
    import pandas
    summary.to_excel('/root/MVdetr/multiview_detector/tracker/OC_SORT/results/eval.xlsx')

    return summary

gt = '/root/autodl-tmp/MultiviewX/mota_gt.txt'
pred = '/root/MVdetr/multiview_detector/tracker/OC_SORT/results/mota_pred.txt'
mot_metrics(pred,gt,scale=0.4)