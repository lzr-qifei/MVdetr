from multiview_detector.evaluation.evaluate import evaluate
import os

gt_path = '/root/vis_results/gt1.txt'
pred_path = '/root/vis_results/test1.txt'
recall, precision, moda, modp = evaluate(pred_path,
                                            gt_path,
                                            'MultiviewX')
print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')