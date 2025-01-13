import numpy as np
import matplotlib.pyplot as plt

class MatrixHistogram:
    def __init__(self):
        # 保存所有矩阵的数值
        self.all_values = []

    def add_matrix(self, matrix):
        """
        接收一个矩阵并将其中的所有值保存到 all_values 中。
        
        参数:
        matrix (2D list 或 np.ndarray): 一个包含距离数值的矩阵
        """
        # self.all_values.extend(matrix.flatten())
        for value in matrix.flatten():
            if value <= 143:
                self.all_values.append(value)

    def save_histogram(self, file_path):
        """
        根据所有保存的数值绘制直方图并将其保存到本地文件。
        
        参数:
        file_path (str): 要保存图像的文件路径
        """
        if not self.all_values:
            print("没有数据可绘制直方图！")
            return
        
        # 将 all_values 转换为 NumPy 数组
        values = np.array(self.all_values)
        # 计算标准差、均值以及标准差除以均值（变异系数）
        std_dev = np.std(values)
        mean = np.mean(values)
        cv = std_dev / mean  # 变异系数：标准差 / 均值
        md = np.median(values)

        # 输出变异系数
        # print(f"当前数据的标准差: {std_dev:.2f}")
        print(f"当前数据的中位数: {md:.2f}")
        print(f"当前数据的均值: {mean:.2f}")
        print(f"当前数据的标准差与均值之比（变异系数）：{cv:.2f}")
        # variance = np.var(values)
        # print(f"当前数据的方差: {variance:.2f}")
        
        # 计算直方图的间隔和范围
        n_bins = 'auto'  # 自动选择合适的间隔
        plt.hist(values, bins=n_bins, edgecolor='black')

        # 设置图形标题和标签
        plt.title('Distribution of Distance Values')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')

        # 保存直方图到本地文件
        plt.savefig(file_path)
        print(f"直方图已保存到 {file_path}")

        # 关闭图形以释放内存
        plt.close()
