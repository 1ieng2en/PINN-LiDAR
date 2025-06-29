import deepxde as dde
import numpy as np
from scipy.spatial.distance import cdist
import torch
import matplotlib.pyplot as plt
from source import create_source
from vis import Avis
from typing import Dict, Tuple, List

class AcousticPINNSampler2D:
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        n_points: int = 20,
        source_params: Dict = None
    ):
        """2D声场采样器"""
        self.bounds = bounds
        self.n_points = n_points
        
        # 初始化声源
        self.source_params = source_params or {
            'freq': 1000,
            'radius': 0.1,
            'amp': 1.0,
            'loc': [0, 0, 0],
            't': 0
        }
        self.source = create_source('spherical', **self.source_params)
        
        # 存储测量数据
        self.measured_points = []
        self.measured_values = []
        
        # 初始化PINN模型
        self.setup_pinn()
        
    def setup_pinn(self):
        """设置PINN模型"""
        # 定义计算域
        self.geom = dde.geometry.Rectangle(
            [self.bounds['x'][0], self.bounds['y'][0]],
            [self.bounds['x'][1], self.bounds['y'][1]]
        )
        
        # 定义PDE，增加权重系数
        def helmholtz_equation(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)
            k = 2 * np.pi * self.source_params['freq'] / 343.0
            # 增加权重系数
            weight = k  # 可以调整这个值
            return weight * (dy_xx + dy_yy + k**2 * y)
        
        self.pde_func = helmholtz_equation

        # 其余代码保持不变
        self.pde = dde.data.PDE(
            self.geom,
            helmholtz_equation,
            [],  # 无边界条件
            num_domain=1000
        )
        # 设置神经网络
        # 增加网络深度和宽度
        layer_size = [2] + [128] * 6 + [2]  # 更深更宽的网络

        # 考虑使用更适合高频特征的激活函数
        activation = "gelu"  # 或者尝试 "gelu"
        initializer = "Glorot uniform"
        
        self.net = dde.nn.FNN(layer_size, activation, initializer)
        
        # 创建模型
        self.model = dde.Model(self.pde, self.net)
    
        # 检查GPU是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # 将模型移到GPU
        if torch.cuda.is_available():
            self.net.cuda()


    def select_next_point(self) -> np.ndarray:
        """选择下一个采样点，基于多个评分标准"""
        n_candidates = 1000
        candidates = []
        while len(candidates) < n_candidates:
            point = np.array([
                np.random.uniform(*self.bounds['x']),
                np.random.uniform(*self.bounds['y'])
            ])
            
            if self.measured_points:
                min_dist = min(np.linalg.norm(point - p[:2]) for p in self.measured_points)
                if min_dist < 0.002:
                    continue
                    
            candidates.append(point)
        
        candidates = np.array(candidates)
        
        # 第一个点随机选择
        if not self.measured_points:
            return candidates[np.random.randint(len(candidates))]
        
        # 1. 空间分布分数：离已有点最远的位置得分高
        spatial_score = np.min(cdist(candidates, np.array(self.measured_points)[:, :2]), axis=1)
        spatial_score = (spatial_score - spatial_score.min()) / (spatial_score.max() - spatial_score.min())
        
        # 2. 预测不确定性分数
        pred = self.model.predict(candidates)
        pred_complex = pred[:, 0] + 1j * pred[:, 1]
        pred_magnitude = np.abs(pred_complex)
        
        # 3. 测量值差异分数：与现有测量点的声压值差异
        measured_magnitudes = np.abs(self.measured_values)
        magnitude_diffs = np.array([np.abs(pred_magnitude - mag) for mag in measured_magnitudes])
        value_score = np.mean(magnitude_diffs, axis=0)  # 平均差异
        value_score = (value_score - value_score.min()) / (value_score.max() - value_score.min())
        
        # 综合评分 (归一化后的加权和)
        total_score = (
            0.1 * spatial_score +    # 空间覆盖
            0.6 * pred_magnitude +   # 预测声压幅值（关注高响应区域）
            0.3 * value_score       # 测量值差异
        )
        
        # 返回得分最高的点
        return candidates[np.argmax(total_score)]
                

    def measure_point(self, point: np.ndarray) -> complex:
        """在给定点进行测量"""
        point_3d = np.append(point, 0)  # 转换为3D点
        value = self.source.compute_pressure(obs=point_3d.reshape(1, -1))[0]

        sp_magnitude = np.abs(value)
        max_magnitude = np.nanmax(sp_magnitude)
        with np.errstate(divide='ignore', invalid='ignore'):
            value = 20 * np.log10(sp_magnitude / max_magnitude)
        
        self.measured_points.append(point_3d)
        self.measured_values.append(value)
        
        return value
        
    def train_model(self):
        """训练PINN模型"""
        if not self.measured_points:
            return
                
        # 准备训练数据
        observe_x = np.array(self.measured_points)[:, :2]
        observe_y = np.array([(v.real, v.imag) for v in self.measured_values])
        
        # 重新创建数据集
        data = dde.data.PDE(
            self.geom,
            self.pde_func,
            [],
            num_domain=1000,
            anchors=observe_x,
            solution=lambda x: observe_y,
            num_test=100
        )
        
        # 重新编译和训练模型
        self.model = dde.Model(data, self.net)
        self.model.compile("adam", lr=0.001, loss="MSE")
                
        from torch.utils.tensorboard import SummaryWriter
        from tebo import TensorBoardCallback
        writer = SummaryWriter('runs/experiment')

        # 创建回调实例并训练
        tb_callback = TensorBoardCallback(writer)
        losshistory, train_state = self.model.train(iterations=10000, callbacks=[tb_callback])
        writer.close()
        


    def compare_fields(self, **kwargs):
        """比较真实声场和PINN预测"""
        grid_params = {
            'grid_size': 100,
            'x_range': (self.bounds['x'][0], self.bounds['x'][1]),
            'y_range': (self.bounds['y'][0], self.bounds['y'][1])
        }
        
        # 设置dB范围
        db_range = kwargs.get('db_range', (-80, 0))  # 默认范围 -80 到 0 dB
        
        # 绘制实际声场
        Avis.create_heat_map(
            self.source,
            grid_params,
            t=self.source_params['t'],
            title="True Field",
            db_range=db_range  # 传递dB范围参数
        )
        
        # 计算PINN预测
        x = np.linspace(*self.bounds['x'], grid_params['grid_size'])
        y = np.linspace(*self.bounds['y'], grid_params['grid_size'])
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.flatten(), Y.flatten()))
        
        pred = self.model.predict(points)
        pressure = pred[:, 0] + 1j * pred[:, 1]
        
        # 用DummySource包装预测结果以使用Avis
        class DummySource:
            def __init__(self, pressure, source_params):
                self.pressure = pressure
                self.freq = source_params['freq']
                self.params = source_params.copy()
                
            def compute_pressure(self, **kwargs):
                obs = kwargs.get('obs')
                if obs is not None:
                    idx = np.where(np.all(points[:, None] == obs[:, :2], axis=2))[0]
                    return self.pressure[idx]
                return self.pressure
        
        dummy_source = DummySource(pressure, self.source_params)
        Avis.create_heat_map(
            dummy_source,
            grid_params,
            t=self.source_params['t'],
            title="PINN Prediction",
            db_range=db_range  # 传递相同的dB范围参数
        )
            
    def run_sampling_sequence(self):
        """执行完整的采样序列"""
        for i in range(self.n_points):
            next_point = self.select_next_point()
            value = self.measure_point(next_point)
            self.train_model()
            
            if (i + 1) % 5 == 0:
                print(f"\n完成 {i+1}/{self.n_points} 个点的采样")
                self.compare_fields()

def run_example():
    bounds = {
        'x': (-2, 2),
        'y': (-2, 2),
    }
    
    source_params = {
        'freq': 1000,
        'radius': 0.1,
        'amp': 1.0,
        'loc': [0, 0, 0],
        't': 0
    }
    
    sampler = AcousticPINNSampler2D(
        bounds=bounds,
        n_points=20,
        source_params=source_params
    )
    
    sampler.run_sampling_sequence()

if __name__ == "__main__":
    run_example()