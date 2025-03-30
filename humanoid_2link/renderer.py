import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import imageio


class MujocoRenderer:
    """
    Class for handling MuJoCo rendering and visualization separately from simulation.
    This handles window creation, scene rendering, and video recording.
    """
    
    def __init__(self, window_size=(1200, 912), title="Ankle Exoskeleton Simulation"):
        """
        Initialize the renderer.
        
        Args:
            window_size: Tuple of (width, height) for the window
            title: Window title
        """
        self.window_size = window_size
        self.title = title
        self.frames = []  # For recording
        self.is_recording = False
        self.window = None
        self.scene = None
        self.context = None
        self.cam = None
        self.opt = None
        
        # Initialize GLFW and create window
        self._init_glfw()
        
    def _init_glfw(self):
        """Initialize GLFW and create a window."""
        glfw.init()
        self.window = glfw.create_window(
            self.window_size[0], 
            self.window_size[1], 
            self.title, 
            None, 
            None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Create camera and options objects
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        
        # Set default camera and option parameters
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
    
    def setup_visualization(self, model, config):
        """
        Configure visualization settings based on config parameters.
        
        Args:
            model: MuJoCo model object
            config: Configuration dictionary with visualization settings
        """
        # Configure visualization flags
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = config.get('visualize_contact_force', False)
        self.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = config.get('visualize_perturbation_force', False)
        self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = config.get('visualize_joints', False)
        self.opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = config.get('visualize_actuators', False)
        self.opt.flags[mj.mjtVisFlag.mjVIS_COM] = config.get('visualize_center_of_mass', False)
        
        # Set camera configuration
        self.cam.azimuth = config.get('camera_azimuth', 90)
        self.cam.distance = config.get('camera_distance', 3.0)
        self.cam.elevation = config.get('camera_elevation', -5)
        
        # Parse and set lookat coordinates
        lookat_string = config.get('camera_lookat_xyz', '0, 0, 1')
        lookat_coords = tuple(map(float, lookat_string.split(', ')))
        self.cam.lookat = np.array([lookat_coords[0], lookat_coords[1], lookat_coords[2]])
        
        # Configure model visualization parameters
        model.vis.map.force = 0.25  # scaling parameter for force vector's length
        model.vis.map.torque = 0.1  # scaling parameter for control torque
        
        # Configure scale parameters
        model.vis.scale.contactwidth = 0.05   # width of the floor contact point
        model.vis.scale.contactheight = 0.01  # height of the floor contact point
        model.vis.scale.forcewidth = 0.03     # width of the force vector
        model.vis.scale.com = 0.2             # com radius
        model.vis.scale.actuatorwidth = 0.1   # diameter of visualized actuator
        model.vis.scale.actuatorlength = 0.1  # thickness of visualized actuator
        model.vis.scale.jointwidth = 0.025    # diameter of joint arrows
        model.vis.scale.framelength = 0.25
        model.vis.scale.framewidth = 0.05
        
        # Configure color parameters
        model.vis.rgba.contactforce = np.array([0.7, 0., 0., 0.5], dtype=np.float32)
        model.vis.rgba.force = np.array([0., 0.7, 0., 0.5], dtype=np.float32)
        model.vis.rgba.joint = np.array([0.2, 1, 0.1, 0.8])
        model.vis.rgba.actuatorpositive = np.array([0., 0.9, 0., 0.5])  # color when actuator is exerting positive force
        model.vis.rgba.actuatornegative = np.array([0.9, 0., 0., 0.5])  # color when actuator is exerting negative force
        model.vis.rgba.com = np.array([1., 0.647, 0., 0.5])  # center of mass color
        
        # Create scene and context
        self.scene = mj.MjvScene(model, maxgeom=10000)
        self.context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    def render(self, model, data):
        """
        Render a single frame of the simulation.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            
        Returns:
            bool: True if rendering should continue, False if window should close
        """
        if self.window_should_close():
            return False
            
        # Get framebuffer viewport size
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        
        # Update scene and render
        mj.mjv_updateScene(
            model, 
            data, 
            self.opt, 
            None, 
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value, 
            self.scene
        )
        mj.mjr_render(viewport, self.scene, self.context)
        
        # Record frame if enabled
        if self.is_recording:
            self._capture_frame(viewport, viewport_height, viewport_width)
            
        # Swap buffers and process events
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
        return True
    
    def _capture_frame(self, viewport, height, width):
        """
        Capture a frame for video recording.
        
        Args:
            viewport: MjrRect viewport object
            height: Viewport height
            width: Viewport width
        """
        rgb_array = np.empty((height, width, 3), dtype=np.uint8)
        depth_array = np.empty((height, width), dtype=np.float32)
        
        mj.mjr_readPixels(
            rgb=rgb_array, 
            depth=depth_array, 
            viewport=viewport, 
            con=self.context
        )
        
        # Flip the image vertically since OpenGL renders with inverted Y axis
        rgb_array = np.flipud(rgb_array)
        self.frames.append(rgb_array)
    
    def start_recording(self):
        """Start recording frames for video."""
        self.is_recording = True
        self.frames = []
    
    def stop_recording(self):
        """Stop recording frames."""
        self.is_recording = False
    
    def save_video(self, filename, fps=60):
        """
        Save recorded frames as a video.
        
        Args:
            filename: Output video filename
            fps: Frames per second
        """
        if self.frames:
            print(f"Saving video to {filename} ({len(self.frames)} frames at {fps} fps)")
            imageio.mimwrite(filename, self.frames, fps=fps)
            self.frames = []
        else:
            print("No frames to save")
    
    def window_should_close(self):
        """Check if the window should close."""
        return glfw.window_should_close(self.window)
    
    def close(self):
        """Clean up resources."""
        # MuJoCo Python bindings handle freeing objects automatically
        # when they go out of scope, so explicit freeing is not needed
            
        glfw.terminate()
        print("Renderer resources released")


# Simple test function to verify the renderer works on its own
if __name__ == "__main__":
    import time
    import yaml
    
    # Load a simple model
    try:
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)['config']
        
        # Create model and data
        model = mj.MjModel.from_xml_path(config['lit_xml_file'])
        data = mj.MjData(model)
        
        # Create renderer
        renderer = MujocoRenderer()
        renderer.setup_visualization(model, config)
        renderer.start_recording()
        
        print("Rendering test scene for 3 seconds...")
        start_time = time.time()
        
        # Simple render loop
        while time.time() - start_time < 3.0:
            # Step physics
            mj.mj_step(model, data)
            
            # Render frame
            if not renderer.render(model, data):
                break
        
        # Save video if recording was enabled
        renderer.save_video("renderer_test.mp4")
        renderer.close()
        print("Renderer test completed successfully!")
        
    except Exception as e:
        print(f"Renderer test failed: {e}")