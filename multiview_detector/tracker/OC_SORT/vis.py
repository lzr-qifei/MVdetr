import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 读取数据，指定逗号为分隔符
data = np.loadtxt('/root/MVdetr/multiview_detector/tracker/OC_SORT/results/pred_40_track_result.txt', delimiter=',')
frame_ids = data[:, 0]
x_coords = data[:, 1]
y_coords = data[:, 2]
ids = data[:, 3]

# 获取唯一的目标ID
unique_ids = np.unique(ids)

# 创建颜色映射
colors = cm.get_cmap('hsv', len(unique_ids))

plt.figure(figsize=(10, 6))
plt.xlim(0, 1000)
plt.ylim(0, 640)

# 按目标ID和时间顺序绘制轨迹
for i, obj_id in enumerate(unique_ids):
    mask = (ids == obj_id)
    # 按时间排序
    sorted_indices = np.argsort(frame_ids[mask])
    plt.plot(x_coords[mask][sorted_indices], y_coords[mask][sorted_indices], color=colors(i), label=f'ID {int(obj_id)}')
    plt.scatter(x_coords[mask][sorted_indices], y_coords[mask][sorted_indices], color=colors(i), s=10)

plt.title('Multi-Object Tracking Results')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid()

# 保存图像
plt.savefig('tracking_results.png', bbox_inches='tight')
plt.close()
