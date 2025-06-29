from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from vis import Avis

class Dataset(ABC):
    def __init__(self, source, n_sensors):
        self.source = source
        self.n_sensors = n_sensors

    @abstractmethod
    def gen(self, method='standard', sampling='random', *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, method = 'standard'):
        pass

    @abstractmethod
    def trunk(self, eval_points):
        pass

    @abstractmethod
    def branch(self, eval_points):
        pass


class Spherical(Dataset):
    """return the pressure reading for dde.data
    input: observation location
    output: pressure"""

    def pde(self, freq, loc):
        "freq: list of frequency"
        
        for f, l in zip(freq, loc):
            self.source.update_params(freq=f, loc=l)
            pressure += self.source.compute_pressure(obs=loc)

        pressure /= f
        return pressure


    def gen(self, method='standard', sampling='random', *args, **kwargs):
        sampling_fn = self.get_sampling_function(sampling)
        eval_points, sensor_data, field_pressure_data = self.compute_pressure(sampling_fn, method=method, **kwargs)
        return self.assemble_matrices(eval_points, sensor_data, field_pressure_data)

    def get_sampling_function(self, sampling):
        sampling_methods = {
            'random': self.random_sampling,
            'grid': self.grid_sampling
        }
        if sampling not in sampling_methods:
            raise ValueError(f"Unknown sampling method: {sampling}")
        return sampling_methods[sampling]

    def compute_pressure(self, sampling_fn, method, *args, **kwargs):
        params = self.get_default_params(method, *args, **kwargs)
        return sampling_fn(*params, *args, **kwargs)

    def get_default_params(self, method, *args, **kwargs):
        if method in ['standard', 'random_ball']:
            return (1000, (100, 1000), (0.05, 0.15), 20)
        else:
            raise ValueError(f"Unknown dataset generation method: {method}")
    
    def random_sampling(self, n_samples, freq_range, radius_range, grid_points, is_random_loc=False, *args, **kwargs):
        eval_points = self.generate_evaluation_points(grid_points)
        np.random.seed(0)
        sensor_data, field_pressure_data = [], []
        for _ in range(n_samples):
            freq_i, radius_i = self.sample_parameters(freq_range, radius_range)
            loc_i = self.sample_location(radius_i) if is_random_loc else np.array([0, 0, 0])
            self.source.update_params(freq=freq_i, radius=radius_i, loc=loc_i)
            sensor_locs = self.generate_sensor_points()
            sensor_pressure = np.fft.fftshift(np.fft.fft(self.source.compute_pressure(obs=sensor_locs)))
            field_pressure = np.fft.fftshift(np.fft.fft(self.source.compute_pressure(obs=eval_points)))
            sensor_data.append((sensor_locs, sensor_pressure, freq_i, radius_i))
            field_pressure_data.append(field_pressure)
        return eval_points, sensor_data, field_pressure_data

    def grid_sampling(self, grid_params, rand_freq_upper=4000, rand_num=10, r_upper=0.1):
        pressure = np.zeros((grid_params['grid_size'], grid_params['grid_size']))
        np.random.seed(21)
        
        # Generate random parameters
        freq = np.random.rand(rand_num) * rand_freq_upper
        loc = [(np.random.rand(3) - 0.5) * (grid_params['x_range'][1] / 2) for _ in range(rand_num)]
        radii = np.random.rand(rand_num) * r_upper

        for f, l in zip(freq, loc):
            self.source.update_params(freq=f, loc=l)
            pressure += Avis.create_heat_map(self.source, grid_params, t=10)

        return None, [(None, None, None, None) for _ in range(grid_params['grid_size'])], pressure.flatten()


    def assemble_matrices(self, eval_points, sensor_data, field_pressure_data):
        """field pressure data: n_samples * block_size
            eval_points: n_samples * 2
            sensor_data: n_samples * branch_dim"""
        block_size = len(field_pressure_data)
        branch_dim = 2 + 3 * self.n_sensors
        n_samples = len(sensor_data)
        bigXb = np.zeros((n_samples * block_size, branch_dim), dtype=np.float32)
        bigXt = np.zeros((n_samples * block_size, 2), dtype=np.float32)
        bigY = np.zeros((n_samples * block_size, 1), dtype=np.float32)

        idx_start = 0
        for i, (sensor_locs, sensor_pressure, freq_i, radius_i) in enumerate(sensor_data):
            if sensor_locs is not None:
                branch_vec = [freq_i, radius_i] + [val for p in zip(sensor_locs[:, 0], sensor_locs[:, 1], sensor_pressure or [0]) for val in p]
                branch_vec = np.array(branch_vec, dtype=np.float32)
                idx_end = idx_start + block_size
                bigXb[idx_start:idx_end] = branch_vec.reshape(1, -1)
                bigXt[idx_start:idx_start + block_size] = eval_points
                bigY[idx_start:idx_start + block_size, 0] = field_pressure_data[i]
                idx_start += block_size

        return bigXb, bigXt, bigY

    def sample_parameters(self, freq_range, radius_range):
        freq_i = np.random.uniform(*freq_range)
        radius_i = np.random.uniform(*radius_range)
        return freq_i, radius_i

    def sample_location(self, radius):
        return (np.random.rand(3) - 0.5) * radius * 10  # Positions randomly scaled by the radius for effect

    def generate_evaluation_points(self, grid_points):
        x_vals = np.linspace(-1, 1, grid_points)
        y_vals = np.linspace(-1, 1, grid_points)
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        eval_points = np.column_stack((Xg.flatten(), Yg.flatten()))
        return eval_points
