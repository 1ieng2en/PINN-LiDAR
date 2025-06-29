# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Avis:
    """Class for handling all visualization tasks for acoustic sources."""
    
    @staticmethod
    def plot_pressure_1d(x, pressure, title = f"Source Pressure"):
        """Plot amplitude, real part, and imaginary part of pressure along a 1D line."""
        amplitude = np.abs(pressure)
        real_part = np.real(pressure)
        imag_part = np.imag(pressure)

        plt.figure(figsize=(5, 4))
        plt.plot(x, amplitude, label='Amplitude', linestyle='-')
        plt.plot(x, real_part, label='Real Part', linestyle='--')
        plt.plot(x, imag_part, label='Imaginary Part', linestyle=':')
        plt.xlabel('Distance (m)')
        plt.ylabel('Pressure')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_frequency_response(freqs, fr, title="Frequency Response", db_ref=1):
        """Plot frequency response magnitude and phase."""
        magnitude_db = 20 * np.log10(np.abs(fr) / db_ref)
        phase_deg = np.angle(fr, deg=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))
        
        ax1.semilogx(freqs, magnitude_db)
        ax1.grid(True)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(f'{title} - Magnitude')
        
        ax2.semilogx(freqs, phase_deg)
        ax2.grid(True)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title(f'{title} - Phase')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spl_heatmap(sp, X, Y, title, db_range=(-80, 0)):
        """Plot the sound pressure level (SPL) heatmap."""
        sp_magnitude = np.abs(sp)
        max_magnitude = np.nanmax(sp_magnitude)
        with np.errstate(divide='ignore', invalid='ignore'):
            spl = 20 * np.log10(sp_magnitude / max_magnitude)
        
        # 限制dB范围
        spl = np.clip(spl, db_range[0], db_range[1])
        
        # plt.figure(figsize=(5, 4))
        plt.imshow(spl, 
                extent=[X.min(), X.max(), Y.min(), Y.max()], 
                origin='lower', 
                cmap='jet',
                vmin=db_range[0],  # 设置最小值
                vmax=db_range[1])  # 设置最大值
        # plt.colorbar(label='Relative Sound Pressure Level (dB)')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(title)
        # plt.show()

    @staticmethod
    def plot_directivity(angles, pressures, title="Directivity Pattern"):
        """Plot directivity pattern in polar coordinates."""
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        pressures_norm = np.abs(pressures) / np.max(np.abs(pressures))
        
        ax.plot(angles, pressures_norm)
        ax.set_title(title)
        ax.grid(True)
        
        ax.set_xticks(np.linspace(0, 2*np.pi - 2*np.pi/12, 12))
        ax.set_xticklabels([f'{int((angle))}°' 
                        for angle in np.linspace(0, 330, 12)])
        ax.set_yticks([0.2, 0.6, 1])
        
        plt.show()

    @staticmethod
    def plot_source_mesh(points, distances, title="Source Mesh"):
        """Plot 3D source mesh with color-coded distances."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(points[:,0], points[:,1], points[:,2], 
                       c=distances, cmap='jet')
        plt.colorbar(sc, label='Distance')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.show()


    @classmethod
    def create_heat_map(cls, source_obj, grid_params, t = 0, **kwargs):
        """Create heat maps for multiple sources."""
        grid_size = grid_params.get('grid_size', 500)
        x_min, x_max = grid_params.get('x_range', (-0.5, 0.5))
        y_min, y_max = grid_params.get('y_range', (-0.5, 0.5))

        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)

        params = source_obj.params
        title = kwargs.get('title', source_obj.__class__.__name__)

        # print(f"\nGenerating heat map for {title}")
        # print("Current parameters:", params)
        
        loc = params.get('loc', [0, 0, 0])
        radius = params.get('radius', 0.05)

        distances = np.sqrt((X - loc[0])**2 + (Y - loc[1])**2)
        mask = distances >= radius
        
        Z = np.zeros_like(X)
        obs = np.column_stack((X[mask].ravel(), Y[mask].ravel(), Z[mask].ravel()))

        params = params.copy()
        params['obs'] = obs
        params.setdefault('t', t)
    
        pressure = source_obj.compute_pressure(**params)

        full_pressure = np.full(X.shape, np.nan)
        if pressure.ndim == 2:
            full_pressure[mask] = pressure[mask]
        else:
            full_pressure[mask] = pressure

        plot_title = f"{title} SPL Heatmap at t = {t}s, f = {source_obj.freq}Hz"
        if not kwargs.get('no_plot', False):
            # 从kwargs获取db_range，默认为(-80, 0)
            db_range = kwargs.get('db_range', (-80, 0))
            cls.plot_spl_heatmap(full_pressure, X, Y, plot_title, db_range=db_range)

        return full_pressure