import mujoco
import glfw
import numpy as np
import time

def main():
    """
    Load and visualize the initial_humanoid.xml model to check exoskeleton visualization.
    
    Controls:
     - ESC: Quit
     - Space: Pause/unpause simulation
     - R: Reset simulation
     - Arrow keys: Move camera
     - Mouse drag: Rotate camera
     - Scroll: Zoom
    """
    # Initialize GLFW
    if not glfw.init():
        return
    
    # Create window
    window = glfw.create_window(1200, 900, "MuJoCo Model Viewer", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Load model from XML file
    model = mujoco.MjModel.from_xml_path("literature_humanoid.xml")
    data = mujoco.MjData(model)
    
    # Create scene and context for rendering
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # Create camera
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    
    # Set camera position for better view of ankle exoskeleton
    camera.lookat[0] = 0
    camera.lookat[1] = 0
    camera.lookat[2] = 0.2
    camera.distance = 1.0
    camera.elevation = -20
    camera.azimuth = 90
    
    # Set visualization options
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    
    # Show site markers if any
    options.flags[mujoco.mjtVisFlag.mjVIS_SDFITER] = True
    
    # For saving frames if needed
    frames = []
    
    # Initialize simulation
    running = True
    paused = False
    
    # Set initial pose to make ankle visible
    data.qpos[0] = 0  # foot orientation
    data.qpos[3] = 0.1  # ankle joint angle (slight dorsiflexion)
    mujoco.mj_forward(model, data)
    
    print("\nControls:")
    print("  ESC: Quit")
    print("  Space: Pause/unpause")
    print("  R: Reset simulation")
    print("  Arrow keys: Move camera")
    print("  Mouse: Rotate/zoom camera")
    
    # Mouse interaction vars
    button_left = False
    button_right = False
    lastx = 0
    lasty = 0
    
    # Mouse button callback
    def mouse_button(window, button, act, mods):
        nonlocal button_left, button_right, lastx, lasty
        
        # Update button state
        button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        
        # Update mouse position
        x, y = glfw.get_cursor_pos(window)
        lastx, lasty = int(x), int(y)
    
    # Mouse move callback
    def mouse_move(window, x, y):
        nonlocal button_left, button_right, lastx, lasty
        
        # Compute mouse displacement
        dx = x - lastx
        dy = y - lasty
        lastx, lasty = int(x), int(y)
        
        # Move camera if button pressed
        if button_left:
            # Rotate camera
            camera.azimuth += dx * 0.3
            camera.elevation += dy * 0.3
        elif button_right:
            # Move camera target
            camera.lookat[0] += dx * 0.01
            camera.lookat[1] += dy * 0.01
    
    # Scroll callback
    def scroll(window, xoffset, yoffset):
        nonlocal camera
        
        # Adjust camera distance (zoom)
        camera.distance *= 0.9 if yoffset > 0 else 1.1
    
    # Key callback
    def key_callback(window, key, scancode, action, mods):
        nonlocal running, paused, data
        
        # ESC: Quit
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            running = False
        
        # Space: Pause/unpause
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            paused = not paused
        
        # R: Reset
        elif key == glfw.KEY_R and action == glfw.PRESS:
            mujoco.mj_resetData(model, data)
            data.qpos[0] = 0  # foot orientation
            data.qpos[3] = 0.1  # ankle joint angle (slight dorsiflexion)
            mujoco.mj_forward(model, data)
            
        # Arrow keys: Move camera target
        elif key == glfw.KEY_UP and action in (glfw.PRESS, glfw.REPEAT):
            camera.lookat[2] += 0.05
        elif key == glfw.KEY_DOWN and action in (glfw.PRESS, glfw.REPEAT):
            camera.lookat[2] -= 0.05
        elif key == glfw.KEY_LEFT and action in (glfw.PRESS, glfw.REPEAT):
            camera.lookat[0] -= 0.05
        elif key == glfw.KEY_RIGHT and action in (glfw.PRESS, glfw.REPEAT):
            camera.lookat[0] += 0.05
            
        # Number keys to change ankle angle
        elif key == glfw.KEY_1 and action == glfw.PRESS:
            data.qpos[3] = -0.3  # plantarflexion
            mujoco.mj_forward(model, data)
        elif key == glfw.KEY_2 and action == glfw.PRESS:
            data.qpos[3] = 0.0  # neutral
            mujoco.mj_forward(model, data)
        elif key == glfw.KEY_3 and action == glfw.PRESS:
            data.qpos[3] = 0.3  # dorsiflexion
            mujoco.mj_forward(model, data)
    
    # Set GLFW callbacks
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)
    glfw.set_key_callback(window, key_callback)
    
    # Timing
    last_update_time = time.time()
    
    # Main loop
    while running and not glfw.window_should_close(window):
        # Get time delta
        current_time = time.time()
        time_delta = current_time - last_update_time
        last_update_time = current_time
        
        # Step simulation if not paused
        if not paused:
            mujoco.mj_step(model, data)
            
            # Oscillate ankle joint to visualize exoskeleton movement
            t = data.time
            data.qpos[3] = 0.3 * np.sin(t * 0.5)  # oscillate between -0.3 and 0.3 radians
            mujoco.mj_forward(model, data)
        
        # Get viewport dimensions
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        # Update scene and render
        mujoco.mjv_updateScene(model, data, options, None, camera, 
                             mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        
        # Process events
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        # Control simulation speed
        time.sleep(max(0, 0.01 - (time.time() - current_time)))
    
    # Clean up
    glfw.terminate()

if __name__ == "__main__":
    main()