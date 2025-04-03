import numpy as np
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue


class Perturbation(ABC):
    """Abstract base class for all perturbation types."""
    
    def __init__(self, enabled=True):
        """
        Initialize the perturbation.
        
        Args:
            enabled: Whether the perturbation is enabled
        """
        self.enabled = enabled
        self.stop_event = threading.Event()
    
    @abstractmethod
    def generate(self, queue):
        """
        Generate perturbation forces and put them in the queue.
        
        Args:
            queue: Queue to put perturbation values into
        """
        pass
    
    def start(self, queue):
        """
        Start the perturbation generation in a separate thread.
        
        Args:
            queue: Queue to put perturbation values into
            
        Returns:
            threading.Thread: The thread generating perturbations
        """
        if not self.enabled:
            return None
            
        self.stop_event.clear()
        thread = threading.Thread(
            target=self.generate,
            args=(queue,),
            daemon=True
        )
        thread.start()
        return thread
    
    def stop(self):
        """Stop the perturbation generation."""
        self.stop_event.set()


class NoPerturbation(Perturbation):
    """No perturbation (always returns zero)."""
    
    def __init__(self):
        super().__init__(enabled=False)
    
    def generate(self, queue):
        """No perturbation to generate."""
        pass
        
    def start(self, queue):
        """No thread to start."""
        return None


class ImpulsePerturbation(Perturbation):
    """Impulse perturbation that applies a force for a short duration at intervals."""
    
    def __init__(self, magnitude=100, duration=0.3, period=3.25, direction=None, enabled=True):
        """
        Initialize impulse perturbation.
        
        Args:
            magnitude: Magnitude of the impulse force
            duration: Duration of each impulse (seconds)
            period: Time between impulses (seconds)
            direction: Direction of the impulse (1 or -1), if None, random direction
            enabled: Whether the perturbation is enabled
        """
        super().__init__(enabled)
        self.magnitude = magnitude
        self.duration = duration
        self.period = period
        self.direction = direction  # If None, random direction
    
    def generate(self, queue):
        """
        Generate impulse perturbations and put them in the queue.
        
        Args:
            queue: Queue to put perturbation values into
        """
        while not self.stop_event.is_set():
            # Wait between impulses
            wait_time = np.random.uniform(self.period, self.period + 1)
            if self.stop_event.wait(wait_time):
                break
            
            # Determine direction
            if self.direction is None:
                direction = np.random.choice([-1, 1])
            else:
                direction = self.direction
            
            # Calculate the force magnitude
            force = direction * self.magnitude
            
            # Start time of the impulse
            start_time = time.time()
            end_time = start_time + self.duration
            
            # Apply the impulse for the duration
            while time.time() < end_time and not self.stop_event.is_set():
                queue.put(force)
                time.sleep(0.001)  # Small sleep to avoid flooding the queue
            
            # Clear the queue after the impulse
            while not queue.empty():
                queue.get()


class SinusoidalPerturbation(Perturbation):
    """Sinusoidal perturbation that applies a continuously varying force."""
    
    def __init__(self, amplitude=50, frequency=0.5, phase=0, enabled=True):
        """
        Initialize sinusoidal perturbation.
        
        Args:
            amplitude: Maximum force magnitude
            frequency: Frequency of the sine wave (Hz)
            phase: Initial phase of the sine wave (radians)
            enabled: Whether the perturbation is enabled
        """
        super().__init__(enabled)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.sample_rate = 0.01  # Time between samples (seconds)
    
    def generate(self, queue):
        """
        Generate sinusoidal perturbations and put them in the queue.
        
        Args:
            queue: Queue to put perturbation values into
        """
        t = 0  # Time variable
        
        while not self.stop_event.is_set():
            # Calculate the force
            force = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
            
            # Put the force in the queue
            queue.put(force)
            
            # Wait for next sample
            if self.stop_event.wait(self.sample_rate):
                break
                
            # Increment time
            t += self.sample_rate


class StepPerturbation(Perturbation):
    """Step perturbation that applies a constant force after a delay."""
    
    def __init__(self, magnitude=80, start_time=5.0, duration=None, enabled=True):
        """
        Initialize step perturbation.
        
        Args:
            magnitude: Magnitude of the step force
            start_time: Time to start applying the force (seconds)
            duration: Duration of the force (seconds), if None, applies until stopped
            enabled: Whether the perturbation is enabled
        """
        super().__init__(enabled)
        self.magnitude = magnitude
        self.start_time = start_time
        self.duration = duration
    
    def generate(self, queue):
        """
        Generate step perturbation and put it in the queue.
        
        Args:
            queue: Queue to put perturbation values into
        """
        # Wait until start time
        if self.stop_event.wait(self.start_time):
            return
        
        # Apply the force
        if self.duration is None:
            # Apply until stopped
            while not self.stop_event.is_set():
                queue.put(self.magnitude)
                time.sleep(0.01)  # Small sleep to avoid flooding the queue
        else:
            # Apply for duration
            end_time = time.time() + self.duration
            while time.time() < end_time and not self.stop_event.is_set():
                queue.put(self.magnitude)
                time.sleep(0.01)  # Small sleep to avoid flooding the queue
        
        # Clear the queue after the perturbation
        while not queue.empty():
            queue.get()


class RandomPerturbation(Perturbation):
    """Random perturbation that applies random forces at random intervals."""
    
    def __init__(self, max_magnitude=100, min_duration=0.2, max_duration=1.0, 
                 min_interval=2.0, max_interval=5.0, enabled=True):
        """
        Initialize random perturbation.
        
        Args:
            max_magnitude: Maximum force magnitude
            min_duration: Minimum duration of each perturbation (seconds)
            max_duration: Maximum duration of each perturbation (seconds)
            min_interval: Minimum time between perturbations (seconds)
            max_interval: Maximum time between perturbations (seconds)
            enabled: Whether the perturbation is enabled
        """
        super().__init__(enabled)
        self.max_magnitude = max_magnitude
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_interval = min_interval
        self.max_interval = max_interval
    
    def generate(self, queue):
        """
        Generate random perturbations and put them in the queue.
        
        Args:
            queue: Queue to put perturbation values into
        """
        while not self.stop_event.is_set():
            # Wait between perturbations
            interval = np.random.uniform(self.min_interval, self.max_interval)
            if self.stop_event.wait(interval):
                break
            
            # Generate random parameters
            magnitude = np.random.uniform(-self.max_magnitude, self.max_magnitude)
            duration = np.random.uniform(self.min_duration, self.max_duration)
            
            # Apply the perturbation
            start_time = time.time()
            end_time = start_time + duration
            
            while time.time() < end_time and not self.stop_event.is_set():
                queue.put(magnitude)
                time.sleep(0.01)  # Small sleep to avoid flooding the queue
            
            # Clear the queue after the perturbation
            while not queue.empty():
                queue.get()


def create_perturbation(config):
    """
    Factory function to create a perturbation instance based on configuration.
    
    Args:
        config: Dictionary containing perturbation configuration
        
    Returns:
        Perturbation: An instance of a Perturbation subclass
    """
    if not config.get('apply_perturbation', False):
        return NoPerturbation()
    
    perturbation_type = config.get('perturbation_type', 'impulse').lower()
    
    if perturbation_type == 'none':
        return NoPerturbation()
    
    elif perturbation_type == 'impulse':
        return ImpulsePerturbation(
            magnitude=config.get('perturbation_magnitude', 100),
            duration=config.get('perturbation_time', 0.3),
            period=config.get('perturbation_period', 3.25),
            direction=config.get('perturbation_direction', None),
            enabled=True
        )
    
    elif perturbation_type == 'sinusoidal':
        return SinusoidalPerturbation(
            amplitude=config.get('perturbation_amplitude', 50),
            frequency=config.get('perturbation_frequency', 0.5),
            phase=config.get('perturbation_phase', 0),
            enabled=True
        )
    
    elif perturbation_type == 'step':
        return StepPerturbation(
            magnitude=config.get('perturbation_magnitude', 80),
            start_time=config.get('perturbation_start_time', 5.0),
            duration=config.get('perturbation_duration', None),
            enabled=True
        )
    
    elif perturbation_type == 'random':
        return RandomPerturbation(
            max_magnitude=config.get('perturbation_max_magnitude', 100),
            min_duration=config.get('perturbation_min_duration', 0.2),
            max_duration=config.get('perturbation_max_duration', 1.0),
            min_interval=config.get('perturbation_min_interval', 2.0),
            max_interval=config.get('perturbation_max_interval', 5.0),
            enabled=True
        )
    
    else:
        print(f"Unknown perturbation type: {perturbation_type}, using None")
        return NoPerturbation()


# For direct testing
if __name__ == "__main__":
    test_queue = Queue()
    
    # Test impulse perturbation
    print("Testing impulse perturbation...")
    impulse = ImpulsePerturbation(magnitude=50, duration=0.5, period=2.0)
    impulse_thread = impulse.start(test_queue)
    
    # Monitor the queue for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        if not test_queue.empty():
            force = test_queue.get()
            print(f"Time: {time.time() - start_time:.2f}, Force: {force}")
        time.sleep(0.1)
    
    # Stop the perturbation
    impulse.stop()
    if impulse_thread:
        impulse_thread.join(timeout=1.0)
    
    print("Test complete")