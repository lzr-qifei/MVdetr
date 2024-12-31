
import motmetrics as mm
import numpy as np

def calculate_hota_det_assa(tSource, gtSource, scale=0.025):
    # 读取 ground truth 和检测结果
    gt = np.loadtxt(gtSource, delimiter=',')
    dt = np.loadtxt(tSource, delimiter=',')
    
    accs = []
    frame_id = 1800
    false_positives_per_frame = []
    
    for seq in np.unique(gt[:, 0]).astype(int):  # 对每个序列进行处理
        acc = mm.MOTAccumulator()  # 初始化一个新的积累器
        
        for frame in np.unique(gt[:, 1]).astype(int):  # 对每一帧处理
            # 获取当前帧的 ground truth 和预测数据
            gt_dets = gt[np.logical_and(gt[:, 0] == seq, gt[:, 1] == frame)][:, (2, 8, 9)]
            dt_dets = dt[np.logical_and(dt[:, 0] == seq, dt[:, 1] == frame)][:, (2, 8, 9)]
            
            # 计算每个框的距离矩阵
            C = mm.distances.norm2squared_matrix(gt_dets[:, 1:3] * scale, dt_dets[:, 1:3] * scale)
            C = np.sqrt(C)  # 将距离转换为欧几里得距离
            
            # 更新 MOTA 累积器
            acc.update(gt_dets[:, 0].astype('int').tolist(),
                       dt_dets[:, 0].astype('int').tolist(),
                       C,
                       frameid=frame)
            
            frame_id += 5  # 每隔5帧处理
            
        accs.append(acc)

    # 创建一个 metrics 对象
    mh = mm.metrics.create()

    # 计算所有累积器的指标
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    
    # 渲染并打印结果
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    print(strsummary)
    
    # 计算 HOTA、DetA 和 AssA 的核心部分
    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives
    TPAs = {}  # Dictionary to store True Positives per target ID
    FNAs = {}  # Dictionary to store False Negatives per target ID
    FPAs = {}  # Dictionary to store False Positives per target ID
    
    all_target_ids = set()  # 存储所有目标 ID
    
    # 通过遍历所有事件来重新计算
    for acc in accs:
        events = acc.events
        # 计算 TP, FN, FP
        TP += len(events[events.Type == 'MATCH'])  # 计算总的 TP
        FN += len(events[events.Type == 'MISS'])  # 计算总的 FN
        FP += len(events[events.Type == 'FP'])  # 计算总的 FP
        
        # 分类计算 TPA, FNA, FPA per target ID
        for _, group in events.groupby('FrameId'):
            for event_type, event_group in group.groupby('Type',observed=False):
                if event_type == 'MATCH':
                    # 对每个匹配，按目标 ID 分类
                    for _, match in event_group.iterrows():
                        target_id = match['OId']
                        all_target_ids.add(target_id)  # 添加到所有目标 ID 集合
                        if target_id not in TPAs:
                            TPAs[target_id] = 0
                            FNAs[target_id] = 0
                            FPAs[target_id] = 0
                        TPAs[target_id] += 1  # 每个目标的匹配数量加 1
                elif event_type == 'MISS':
                    # 对每个漏检，按目标 ID 分类
                    for _, miss in event_group.iterrows():
                        target_id = miss['HId']
                        all_target_ids.add(target_id)  # 添加到所有目标 ID 集合
                        if target_id not in FNAs:
                            FNAs[target_id] = 0
                        FNAs[target_id] += 1  # 每个目标的漏检数量加 1
                elif event_type == 'FP':
                    # 对每个误检，按目标 ID 分类
                    for _, fp in event_group.iterrows():
                        target_id = fp['HId']
                        all_target_ids.add(target_id)  # 添加到所有目标 ID 集合
                        if target_id not in FPAs:
                            FPAs[target_id] = 0
                        FPAs[target_id] += 1  # 每个目标的误检数量加 1
    
    # 计算 \mathcal{A}(c) = TP / (TP + FN + FP) for each target ID (c)
    A = {target_id: TPAs.get(target_id, 0) / (TPAs.get(target_id, 0) + FNAs.get(target_id, 0) + FPAs.get(target_id, 0))
         if (TPAs.get(target_id, 0) + FNAs.get(target_id, 0) + FPAs.get(target_id, 0)) > 0
         else 0 for target_id in all_target_ids}
    
    # 计算 DetA
    DetA = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0
    
    # 计算 AssA
    AssA = np.mean(list(A.values())) if TP > 0 else 0
    
    # 计算 HOTA = sqrt(DetA * AssA)
    HOTA = np.sqrt(DetA * AssA) if (DetA * AssA) > 0 else 0
    
    # 输出结果
    print(f'HOTA = {HOTA}')
    print(f'DetA = {DetA}')
    print(f'AssA = {AssA}')





# 示例调用
gt = '/share2/dataset/MultiviewX/mota_gt.txt'  # 你的真值文件路径
pred = '/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/results/mota_pred.txt'  # 你的预测结果文件路径
# gt = '/home/lizirui/det_results/aa.txt'
# pred = '/home/lizirui/MVdetr/multiview_detector/tracker/OC_SORT/wild_results/mota_pred.txt'
calculate_hota_det_assa(gt, pred,1)
