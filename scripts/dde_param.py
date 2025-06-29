# acoustic_deeponet.py
import numpy as np
import deepxde as dde
import torch
import matplotlib.pyplot as plt
from source import create_source
from vis import Avis
from dataset import Dataset



class AcousticDeepONet():
    def __init__(self, n_sensors=20, dataset_cls=Dataset):
        """
        :param n_sensors: 传感器数量（每个样本都会随机生成 n_sensors 个坐标并采样其声压）
        """
        self.params = {}
        self.n_sensors = n_sensors
        self.source = None
        self.net = None  # 存放 DeepONet 神经网络结构
        self.dataset_cls = dataset_cls

    def setup_source(self, source_type='spherical', **params):
        """
        设置使用何种声源类型，以及它的默认参数。
        """
        self.source = create_source(source_type, **params)
        return self

    def create_net(self, hidden_dim=40):
        """
        深度算子网络 (DeepONet) 推荐的写法：
          Branch_net: 输入维度 = 2 (freq, radius) + 3*n_sensors (每个传感器的 x,y,p)
          Trunk_net:  输入维度 = 2 (要预测场点的 x,y)

        这里我们把 Branch_net 的第一层大小改为 [2 + 3*n_sensors].
        """
        # 分支网络的输入维度
        branch_input_dim = 2 + 3 * self.n_sensors  # freq, radius + (x, y, p)*n_sensors
        trunk_input_dim = 2                       # 场点坐标 (x, y)

        net = dde.nn.DeepONet(
            # Branch: [ (2 + 3*n_sensors), 80, 80 ]
            [branch_input_dim, 80, 80],
            # Trunk: [2, 80, 80]
            [trunk_input_dim, 80, 80],
            "relu",               # activation
            "Glorot normal",      # initializer
        )

        return net
    
    
    def branch_data(self, method='standard', *args, **kwargs):
        generator = self.dataset_cls(self.source, self.n_sensors)
        return generator.branch_data(method=method, *args, **kwargs)
    
    


    def generate_sensor_points(self, r_range=(0.15, 1.0)):
        """
        随机在环状区域 [r_range[0], r_range[1]] 中生成 n_sensors 个点 (x, y, z=0)。
        """
        theta = np.random.uniform(0, 2 * np.pi, self.n_sensors)
        r = np.random.uniform(*r_range, self.n_sensors)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(x)
        return np.column_stack((x, y, z))

    def generate_dataset(
        self,
        n_samples=1000,
        grid_points=20,
        freq_range=(100, 1000),
        radius_range=(0.05, 0.15),
        seed=0
    ):
        """
        一次性生成 `n_samples` 条样本。

        对每条样本：
          - 随机生成 freq, radius
          - 随机生成 n_sensors 个 (x_i, y_i) 传感器点
          - 分别计算传感器处的声压 p_i
          - 作为 Branch 网络的输入: [freq, radius, x_1, y_1, p_1, ..., x_n, y_n, p_n]
          - 同时在 grid_points^2 坐标点上计算场值(标签)
          - 对每个场点 (x, y) 都要重复这同一个 Branch 向量

        返回:
          Xb.shape = (n_samples * grid_points^2,  2 + 3*n_sensors )
          Xt.shape = (n_samples * grid_points^2,  2)
          y.shape  = (n_samples * grid_points^2,  1)
        """
        np.random.seed(seed)

        # 在 -2~2 区域生成一个 grid_points x grid_points 的网格（Trunk坐标）
        x_vals = np.linspace(-1, 1, grid_points)
        y_vals = np.linspace(-1, 1, grid_points)
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        eval_points = np.column_stack((Xg.flatten(), Yg.flatten(), np.zeros(Xg.size)))  # (grid_points^2, 3)

        # 准备拼接大矩阵
        block_size = grid_points * grid_points  # 每条样本对应多少个场点
        branch_dim = 2 + 3 * self.n_sensors     # freq, radius + (x,y,p)*n_sensors

        bigXb = np.zeros((n_samples * block_size, branch_dim), dtype=np.float32)
        bigXt = np.zeros((n_samples * block_size, 2), dtype=np.float32)
        bigY  = np.zeros((n_samples * block_size, 1), dtype=np.float32)

        idx_start = 0
        for i in range(n_samples):
            # 1) 随机生成源参数 freq, radius
            freq_i = np.random.uniform(*freq_range)
            radius_i = np.random.uniform(*radius_range)

            # 更新 source 的参数
            self.source.freq = freq_i
            self.source.update_params(freq=freq_i, radius=radius_i)

            # 2) 随机生成传感器点 & 计算声压
            sensor_locs = self.generate_sensor_points()
            sensor_pressure = (self.source.compute_pressure(obs=sensor_locs))  # shape=(n_sensors,)

            # 3) 计算网格上场值(用来做标签)
            field_pressure = (self.source.compute_pressure(obs=eval_points))   # shape=(block_size,)

            # fft场值
            if sensor_pressure.ndim == 2:
                sensor_pressure = np.fft.fftshift(np.fft.fft2(sensor_pressure))
                field_pressure = np.fft.fftshift(np.fft.fft2(field_pressure))
            else:
                sensor_pressure = np.fft.fftshift(np.fft.fft(sensor_pressure))
                field_pressure = np.fft.fftshift(np.fft.fft(field_pressure))

            # 4) 构造本条样本对应的 Branch 向量
            #    branch_vec = [freq, radius, x_1, y_1, p_1, ..., x_n, y_n, p_n]
            branch_vec = []
            branch_vec.append(freq_i)
            branch_vec.append(radius_i)
            for j in range(self.n_sensors):
                branch_vec.append(sensor_locs[j, 0])   # x_j
                branch_vec.append(sensor_locs[j, 1])   # y_j
                branch_vec.append(sensor_pressure[j])  # p_j

            branch_vec = np.array(branch_vec, dtype=np.float32)

            # 5) 填充到 bigXb, bigXt, bigY
            idx_end = idx_start + block_size

            # bigXb: 对 block_size 个网格点，都使用同一个 branch_vec
            #        故需要 repeat
            bigXb[idx_start:idx_end] = branch_vec.reshape(1, -1)

            # bigXt: 就是网格坐标 (x, y)
            bigXt[idx_start:idx_end] = eval_points[:, :2]

            # bigY: 网格上实际的声压场值
            bigY[idx_start:idx_end, 0] = field_pressure

            idx_start = idx_end

        return bigXb, bigXt, bigY

    def positional(data, freq = 1000.0):
        """
        Positional encoding function to add position information to the input data.
        
        Parameters:
        data (ndarray): The input data to encode with dimensions (batch_size, max_len, d_model).
        
        Returns:
        ndarray: The data with added positional encoding.
        """
        # Determine max_len and d_model from the input data
        max_len, d_model = data.shape
        
        # Create a matrix of positions and dimensions
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(freq) / d_model))
        
        # Compute sin and cos for even and odd indices
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Add positional encoding to the data
        data_with_pe = data + pe
        
        return data_with_pe, pe
    

    def train(
        self,
        data,
        n_train=800,
        n_test=200,
        grid_points=20,
        epochs=1000,
        display_every=100
    ):
        """
        用“DeepXDE推荐”的写法：一次性生成数据 -> 交给 model -> 直接 train(epochs=..., ...)

        :param n_train: 训练样本条数
        :param n_test:  测试样本条数
        :param grid_points: 网格分辨率 (即 trunk 上的场点数量)
        :param epochs: 训练总epoch数
        :param display_every: 每多少步打印一次进度
        """

        # 4) 建立 model 并训练
        model = dde.Model(data, self.net)
        model.compile("adam", lr=0.001)

        print(f"\nStart training with epochs={epochs}, display_every={display_every} ...")
        losshistory, train_state = model.train(epochs=epochs,
                                               display_every=display_every,
                                               model_save_path="para_m/final")  # 这里可设置保存路径

        return model, losshistory, train_state

    def predict_field(
        self,
        model,
        freq=1000,
        radius=0.5,
        grid_size=50,
        r_range=(0.2, 1.0)
    ):
        """
        用训练好的 model 在一张新的 (grid_size x grid_size) 网格上预测声场。

        1) 随机生成 n_sensors 个传感器点 (branch 输入)，并更新声源 freq, radius
        2) 计算这些传感器的声压
        3) 把 [freq, radius, x_i, y_i, p_i, ...] 组成 Branch 向量
        4) 生成 (grid_size x grid_size) 网格的 (x,y)，作为 Trunk 输入
        5) 做预测

        返回：
          Xg, Yg, pred_field (后两个可用于画 contour 或 heatmap)
        """
        if model is None:
            raise ValueError("Need a trained dde.Model for prediction.")

        # 1) 随机传感器 + 更新源
        sensor_locs = self.generate_sensor_points(r_range=r_range)
        self.source.freq = freq
        self.source.update_params(freq=freq, radius=radius)

        sensor_pressure = (self.source.compute_pressure(obs=sensor_locs))
        
        if sensor_pressure.ndim == 2:
            sensor_pressure = np.fft.fft2(np.fft.fftshift(sensor_pressure))
        else:
            sensor_pressure = np.fft.fft(np.fft.fftshift(sensor_pressure))

        # 2) Branch 向量: [freq, radius, x_1, y_1, p_1, ..., x_n, y_n, p_n]
        branch_vec = [freq, radius]
        for i in range(self.n_sensors):
            branch_vec.append(sensor_locs[i, 0])
            branch_vec.append(sensor_locs[i, 1])
            branch_vec.append(sensor_pressure[i])
        branch_vec = np.array(branch_vec, dtype=np.float32)

        self.branch_vec = branch_vec

        # 3) 生成网格 trunk
        x_vals = np.linspace(-1, 1, grid_size)
        y_vals = np.linspace(-1, 1, grid_size)
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        coords = np.column_stack((Xg.flatten(), Yg.flatten(), np.zeros(Xg.size)))  # (grid_size^2,3)

        # 4) 组装 DeepONet 输入: (Xb_pred, Xt_pred)
        #    - Xt_pred = 网格坐标 (x, y)
        #    - Xb_pred = 对网格上每个点，都重复同一个 branch_vec
        n_pts = coords.shape[0]
        Xt_pred = coords[:, :2].astype(np.float32)

        Xb_pred = np.repeat(branch_vec.reshape(1, -1), n_pts, axis=0).astype(np.float32)

        # 5) 前向预测
        pred = model.predict((Xb_pred, Xt_pred))  # shape=(n_pts,1)
        pred = pred.reshape(grid_size, grid_size)

        # pred = np.fft.ifft2(np.fft.ifftshift(pred))

        return Xg, Yg, pred

    def compare_true_pred_field(self, model, freq, radius, grid_size=50):
        """
        做一个简单对比：用随机传感器 + freq, radius，
        先拿 model.predict 得到预测场，再用 self.source.compute_pressure(obs=网格点) 得到真实场。
        """
        grid_params = {
            'grid_size': grid_size,
            'x_range': (-2, 2),
            'y_range': (-2, 2)
        }

        # 先预测
        Xg, Yg, pred_field = self.predict_field(model, freq=freq, radius=radius, grid_size=grid_size)

        pred_field = np.fft.ifft2(np.fft.ifftshift(pred_field))

        # 用一个 DummySource 包装预测结果以使用 Avis 的可视化函数
        class DummySource:
            def __init__(self, pressure, source_params):
                self.pressure = pressure
                self.freq = source_params['freq']
                self.params = source_params.copy()
            def compute_pressure(self, **kwargs):
                return self.pressure

        dummy_source = DummySource(pred_field, self.source.params)

        # 再计算真实场
        plt.figure(figsize=(12, 8))
        plt.subplot(2,2,1)
        Avis.create_heat_map(self.source, grid_params, t = 0)
        plt.title("True Field")
        plt.subplot(2,2,2)
        Avis.create_heat_map(dummy_source, grid_params, t = 0)
        plt.title("Predicted Field")
        plt.show()

        return Xg, Yg, pred_field


        # 再计算真实场
        plt.figure(figsize=(12, 8))
        plt.subplot(2,2,1)
        # 再计算真实场
        plt.title("True Field")
        plt.subplot(2,2,2)
        Avis.create_heat_map(dummy_source, grid_params, t = 0)
        plt.title("Predicted Field")
        plt.show()

        return Xg, Yg, pred_field

