# import mujoco as mj
# from mujoco.glfw import glfw
import numpy as np
# import os
# from typing import Callable, Optional, Union, List
# import scipy.linalg
# import mediapy as media
# import matplotlib.pyplot as plt
import threading
import time
from queue import Queue
# import xml.etree.ElementTree as ET
# import imageio
import matplotlib.pyplot as plt

def plot_lists(x_values, y_values1, label='List'):
    """
    Plot two lists as x-y data.

    Parameters:
    - x_values: List of x-axis values.
    - y_values1: List of y-axis values for the first data set.
    - y_values2: List of y-axis values for the second data set.
    - label1: Label for the first data set (default: 'List 1').
    - label2: Label for the second data set (default: 'List 2').
    """
    # Plotting the first list
    plt.plot(x_values, y_values1, label=label)

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()

def generate_large_impulse(time_queue, perturbation_queue, impulse_time):
    
    test_time = 10
    test_start_time = time.time()
    while (time.time()-test_start_time) < test_time:

        wait_time = np.random.uniform(2, 3)
        time.sleep(wait_time)
        
        # Generate a large impulse
        perturbation = np.random.uniform(30, 50)

        start_time = time.time()

        while (time.time() - start_time) < impulse_time:

            # Put the generated impulse into the result queue
            perturbation_queue.put(perturbation)
            time_queue.put(time.time()-test_start_time)

if __name__ == "__main__":

    perturbation_queue = Queue()
    time_queue = Queue()
    impulse_time = 1
    perturbation_list = []
    time_list = []

    impulse_thread = threading.Thread \
                    (target=generate_large_impulse,
                    args=(time_queue, perturbation_queue, impulse_time,) )
    impulse_thread.start()

    start_time = time.time()
    
    while time.time() - start_time < 11:
    
        if not perturbation_queue.empty():
            perturbation_list.append(perturbation_queue.get())
            time_list.append(time_queue.get())
        else:
            perturbation_list.append(0)
            time_list.append(time.time()-start_time)

    
    plot_lists(time_list, perturbation_list, label="Impulse versus time")





