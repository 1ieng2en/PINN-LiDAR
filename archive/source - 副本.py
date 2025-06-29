# acoustic_sources.py
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import spherical_jn, spherical_yn, lpmv
from vis import Avis

class Source(ABC):
    """Base class for acoustic sources."""
    
    def __init__(self, freq=1000, c=343, **kwargs):
        """Initialize source with common parameters."""
        self.params = {
            'freq': freq,
            'c': c,
            **kwargs
        }
        self._update_derived_params()
    
    def _update_derived_params(self):
        """Update derived parameters based on current frequency."""
        self.params['omega'] = 2 * np.pi * self.params['freq']
        self.params['k'] = self.params['omega'] / self.params['c']
    
    def update_params(self, **kwargs):
        """Update parameters and recalculate derived values."""
        self.params.update(kwargs)
        if 'freq' in kwargs:
            self._update_derived_params()
        print("\nCurrent parameters:")
        for key, value in self.params.items():
            print(f"{key}: {value}")
    
    @abstractmethod
    def compute_pressure(self, **kwargs):
        """Compute pressure field."""
        pass
    
    def calculate_frequency_response(self, freqs, obs_point, **kwargs):
        """Calculate frequency response at observation point."""
        obs = np.array([obs_point])
        fr = np.zeros(len(freqs), dtype=complex)
        
        print("\nCalculating frequency response with parameters:")
        print(f"Observation point: {obs_point}")
        print("Additional parameters:", kwargs)
        
        for i, f in enumerate(freqs):
            kwargs['freq'] = f
            kwargs['obs'] = obs
            pressure = self.compute_pressure(**kwargs)
            fr[i] = pressure[0]
        
        return fr
    
    def plot_directivity_pattern(self, observation_radius=1.0, angles=None, **kwargs):
        """Plot directivity pattern."""
        if angles is None:
            angles = np.linspace(0, 2*np.pi, 360)
        
        obs_points = []
        for theta in angles:
            x = observation_radius * np.cos(theta)
            z = observation_radius * np.sin(theta)
            obs_points.append([x, 0, z])
        obs_points = np.array(obs_points)
        
        kwargs['obs'] = obs_points
        print("\nCalculating directivity pattern with parameters:")
        print(f"Observation radius: {observation_radius}")
        print("Additional parameters:", kwargs)
        
        pressures = self.compute_pressure(**kwargs)
        
        title = f'{self.__class__.__name__} Directivity Pattern\nf = {kwargs.get("freq", self.params["freq"])} Hz'
        Avis.plot_directivity(angles, pressures, title)

class SphericalSource(Source):
    def compute_pressure(self, obs, amp=1.0, loc=[0,0,0], radius=0.05, max_order=10, t=None, **kwargs):
        """Compute spherical source pressure."""
        freq = kwargs.get('freq', self.params['freq'])
        omega = 2 * np.pi * freq
        k = omega / self.params['c']
        
        loc = np.array(loc)
        obs = np.array(obs)
        obs_rel = obs - loc
        
        r_obs = np.linalg.norm(obs_rel, axis=1)
        theta_obs = np.arccos(np.clip(obs_rel[:, 2] / r_obs, -1.0, 1.0))
        
        sp = np.zeros_like(r_obs, dtype=complex)
        
        for n in range(max_order + 1):
            jn_ka = spherical_jn(n, k * radius)
            hn_kr = spherical_jn(n, k * r_obs) - 1j * spherical_yn(n, k * r_obs)
            
            if jn_ka == 0:
                continue
                
            An = amp * (2 * n + 1) * jn_ka
            Pn_cos_theta = lpmv(0, n, np.cos(theta_obs))
            sp += An * hn_kr * Pn_cos_theta
        
        if t is not None:
            sp *= np.exp(-1j * omega * t)
            
        sp /= r_obs
        return sp


class DipoleSource(Source):
    def compute_pressure(self, obs, amp=1.0, loc=[0,0,0], sep=0.1, t=None, **kwargs):
        """Compute dipole source pressure."""
        freq = kwargs.get('freq', self.params['freq'])
        omega = 2 * np.pi * freq
        k = omega / self.params['c']
        
        loc = np.array(loc)
        sep_vector = np.array([sep / 2, 0, 0])
        loc1 = loc - sep_vector
        loc2 = loc + sep_vector
        
        obs = np.array(obs)
        r1 = np.linalg.norm(obs - loc1, axis=1)
        r2 = np.linalg.norm(obs - loc2, axis=1)
        
        sp1 = amp / r1
        sp2 = -amp / r2  # Opposite amplitude
        
        if t is not None:
            phase1 = -k * r1 + omega * t
            phase2 = -k * r2 + omega * t
        else:
            phase1 = -k * r1
            phase2 = -k * r2
        
        pressure = sp1 * np.exp(1j * phase1) + sp2 * np.exp(1j * phase2)
        return pressure

class CylinderSource(Source):
    def __init__(self, freq=1000, c=343, baffle_type='free_field', **kwargs):
        """Initialize cylinder source with baffle type."""
        super().__init__(freq=freq, c=c, **kwargs)
        self.params['baffle_type'] = baffle_type
    
    def set_baffle(self, baffle_type):
        """Set the baffle type."""
        if baffle_type not in ['free_field', 'infinite_baffle']:
            raise ValueError("Baffle type must be 'free_field' or 'infinite_baffle'")
        self.params['baffle_type'] = baffle_type
        print(f"\nBaffle type set to: {baffle_type}")
    
    def compute_pressure(self, obs, radius=0.05, length=1.0, amp=1.0, t=None, **kwargs):
        """Compute cylinder source pressure."""
        freq = kwargs.get('freq', self.params['freq'])
        omega = 2 * np.pi * freq
        k = omega / self.params['c']
        
        obs = np.array(obs)
        
        if self.params['baffle_type'] == 'free_field':
            r = np.linalg.norm(obs[:, :2], axis=1) - radius
            r = np.maximum(r, 1e-6)  # Avoid negative/zero distances
            
            sp = amp / np.sqrt(r)
            
            if t is not None:
                phase = -k * r + omega * t
            else:
                phase = -k * r
            
            pressure = sp * np.exp(1j * phase)
            
        elif self.params['baffle_type'] == 'infinite_baffle':
            # Direct sound
            r_direct = np.linalg.norm(obs[:, :2], axis=1)
            r_direct = np.maximum(r_direct, 1e-6)
            
            sp_direct = amp / np.sqrt(r_direct)
            if t is not None:
                phase_direct = -k * r_direct + omega * t
            else:
                phase_direct = -k * r_direct
            pressure_direct = sp_direct * np.exp(1j * phase_direct)
            
            # Image source
            obs_image = obs.copy()
            obs_image[:, 2] = -obs_image[:, 2]
            
            r_image = np.linalg.norm(obs_image[:, :2], axis=1)
            r_image = np.maximum(r_image, 1e-6)
            
            sp_image = amp / np.sqrt(r_image)
            if t is not None:
                phase_image = -k * r_image + omega * t
            else:
                phase_image = -k * r_image
            pressure_image = sp_image * np.exp(1j * phase_image)
            
            pressure = pressure_direct + pressure_image
            
        return pressure

def create_source(source_type, **kwargs):
    """Factory function to create acoustic sources."""
    source_classes = {
        'spherical': SphericalSource,
        'dipole': DipoleSource,
        'cylinder': CylinderSource
    }
    
    if source_type not in source_classes:
        raise ValueError(f"Unknown source type: {source_type}. Available types: {list(source_classes.keys())}")
    
    return source_classes[source_type](**kwargs)


##################################### Tests#####################################

if __name__ == "__main__":
    import numpy as np
    from vis import Avis
    
    def test_frequency_response():
        """Test frequency response calculation and visualization for all sources."""
        print("\n=== Testing Frequency Response ===")
        
        # Create sources
        sources = {
            'spherical': create_source('spherical', freq=1000, radius=0.05),
            'dipole': create_source('dipole', freq=1000, sep=0.1),
            'cylinder': create_source('cylinder', freq=1000)
        }
        
        # Setup frequency response parameters
        freqs = np.logspace(1, 4, 1000)  # 10 Hz to 10 kHz
        obs_point = np.array([1, 0, 0])  # Observation point on x-axis
        
        # Calculate and plot frequency response for each source
        for name, source in sources.items():
            print(f"\nCalculating frequency response for {name} source...")
            fr = source.calculate_frequency_response(
                freqs=freqs,
                obs_point=obs_point,
                amp=1.0,
                loc=[0, 0, 0]
            )
            Avis.plot_frequency_response(freqs, fr, f"{name.title()} Source")
    
    def test_directivity_patterns():
        """Test directivity pattern calculation and visualization."""
        print("\n=== Testing Directivity Patterns ===")
        
        # Create sources with different parameters
        sources = {
            'spherical': create_source('spherical', freq=2000, radius=0.1),
            'dipole': create_source('dipole', freq=2000, sep=0.2),
            'cylinder': create_source('cylinder', freq=2000, baffle_type='infinite_baffle')
        }
        
        # Plot directivity patterns
        for name, source in sources.items():
            print(f"\nPlotting directivity pattern for {name} source...")
            source.plot_directivity_pattern(
                observation_radius=2.0,
                amp=1.0,
                loc=[0, 0, 0]
            )
    
    def test_heat_maps():
        """Test heat map generation for all sources."""
        print("\n=== Testing Heat Maps ===")
        
        # Create sources
        sps = create_source('spherical', freq=3000)
        dps = create_source('dipole', freq=3000)
        cyl = create_source('cylinder', freq=3000)
        
        # Define grid parameters
        grid_params = {
            'grid_size': 200,  # Reduced for faster testing
            'x_range': (-2, 2),
            'y_range': (-2, 2)
        }
        
        # Define sources and their parameters for heat map
        sources = [
            (sps, {'radius': 0.1, 'amp': 1.0, 'loc': [0, 0, 0], 'max_order': 10}, 'Spherical Source'),
            (dps, {'amp': 1.0, 'loc': [0, 0, 0], 'sep': 0.2}, 'Dipole Source'),
            (cyl, {'radius': 0.1, 'length': 1.0, 'amp': 1.0}, 'Cylinder Source')
        ]
        
        # Generate heat maps
        print("\nGenerating heat maps...")
        Avis.create_heat_map(sources, grid_params, freq=3000, t=0)
    
    def test_parameter_updating():
        """Test parameter updating functionality."""
        print("\n=== Testing Parameter Updates ===")
        
        source = create_source('spherical', freq=1000, radius=0.05)
        print("\nInitial parameters:")
        print(source.params)
        
        print("\nUpdating parameters...")
        source.update_params(freq=2000, radius=0.1, amp=2.0)
        print("\nUpdated parameters:")
        print(source.params)
    
    def test_cylinder_baffle():
        """Test cylinder source baffle switching."""
        print("\n=== Testing Cylinder Baffle Types ===")
        
        cyl = create_source('cylinder', freq=2000)
        
        # Test free field
        print("\nTesting free field configuration...")
        cyl.set_baffle('free_field')
        
        # Test infinite baffle
        print("\nTesting infinite baffle configuration...")
        cyl.set_baffle('infinite_baffle')
        
        # Test invalid baffle type
        print("\nTesting invalid baffle type...")
        try:
            cyl.set_baffle('invalid_type')
        except ValueError as e:
            print(f"Caught expected error: {e}")
    
    # Run all tests
    def run_all_tests():
        """Run all test functions."""
        print("Starting acoustic source tests...")
        
        test_parameter_updating()
        test_frequency_response()
        test_directivity_patterns()
        test_heat_maps()
        test_cylinder_baffle()
        
        print("\nAll tests completed successfully!")
    
    # Execute all tests if run directly
    run_all_tests()