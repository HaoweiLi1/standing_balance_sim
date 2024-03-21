import matplotlib.pyplot as plt

def plot_3d_pose_trajectory(positions, orientations):
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_title('$Body\;Center\;of\;Mass\;Trajectory}$')

    # Plot the trajectory
    ax.plot(positions[1:, 1], positions[1:, 2], positions[1:, 3], marker=".",color='k')

    # Draw the orientation vectors
    for i in range(1,len(positions),10):
        ax.quiver(positions[i, 1], positions[i, 2], positions[i, 3],
                  orientations[i][0], orientations[i][3], orientations[i][6],
                  color='r', length=0.1)
        
        ax.quiver(positions[i, 1], positions[i, 2], positions[i, 3],
                  orientations[i][1], orientations[i][4], orientations[i][7],
                  color='g', length=0.1)

        ax.quiver(positions[i, 1], positions[i, 2], positions[i, 3],
                  orientations[i][2], orientations[i][5], orientations[i][8],
                  color='b', length=0.1)
    ax.set_ylim(-0.5,0.5)
    legend_entries = ["COM Pos.", "$\\theta_x$", "$\\theta_y$", "$\\theta_z$"]
    # plt.legend(labels=legend_entries)
    plt.show()

def plot_columns(data_array, y_axis):
    """
    Plot the data in the first column versus the data in the second column.

    Parameters:
    - data_array: NumPy array with at least two columns.

    Returns:
    - None
    """
    # Check if the array has at least two columns
    if data_array.shape[1] < 2:
        print("Error: The input array must have at least two columns.")
        return

    # Extract the columns for plotting
    x_values = data_array[1:, 0]
    y_values = data_array[1:, 1]

    # Plot the data
    plt.plot(x_values, y_values, linestyle='-', color='b', label=y_axis)

    # Add labels and a title
    plt.xlabel('$\\t{Time [sec]}$')
    plt.ylabel(y_axis)
    plt.title(y_axis + "$\\bf{,\;versus\;Time, \\it{t}}$")

    # Add a legend
    # plt.legend()

    # Show the plot
    plt.show()

def plot_two_columns(data_array1, data_array2, y_axis1, y_axis2):
    """
    Plot the data in the first column of data_array1 versus the data in the first column of data_array2,
    and the second column of data_array1 versus the second column of data_array2.

    Parameters:
    - data_array1: NumPy array with at least two columns.
    - data_array2: NumPy array with at least two columns.
    - y_axis1: Label for the y-axis of the first plot.
    - y_axis2: Label for the y-axis of the second plot.

    Returns:
    - None
    """
    # Check if the arrays have at least two columns
    if data_array1.shape[1] < 2 or data_array2.shape[1] < 2:
        print("Error: Both input arrays must have at least two columns.")
        return

    # Extract the columns for plotting
    x_values1 = data_array1[1:, 0]
    y_values1 = data_array1[1:, 1]

    x_values2 = data_array2[1:, 0]
    y_values2 = data_array2[1:, 1]

    # Plot the data for the first array
    plt.plot(x_values1, y_values1, linestyle='-', color='b', label=y_axis1)

    # Plot the data for the second array on the same plot
    plt.plot(x_values1, y_values2, linestyle='-', color='r', label=y_axis2)

    # Add labels and a title
    plt.xlabel('Time [sec]')
    plt.title(f"{y_axis1} and {y_axis2} versus Time")

    # Add legends
    plt.legend()

    # Show the plot
    plt.show()

def plot_four_columns(data_array1, data_array2, data_array3, data_array4, y_axis1, y_axis2, y_axis3, y_axis4):
    """
    Plot the data in the first column of each data array versus the data in the second column for all four datasets.

    Parameters:
    - data_array1: NumPy array with at least two columns.
    - data_array2: NumPy array with at least two columns.
    - data_array3: NumPy array with at least two columns.
    - data_array4: NumPy array with at least two columns.
    - y_axis1: Label for the y-axis of the first plot.
    - y_axis2: Label for the y-axis of the second plot.
    - y_axis3: Label for the y-axis of the third plot.
    - y_axis4: Label for the y-axis of the fourth plot.

    Returns:
    - None
    """
    # Check if the arrays have at least two columns
    if (
        data_array1.shape[1] < 2 or
        data_array2.shape[1] < 2 or
        data_array3.shape[1] < 2 or
        data_array4.shape[1] < 2
    ):
        print("Error: All input arrays must have at least two columns.")
        return

    # Extract the columns for plotting
    x_values = data_array1[1:, 0]

    y_values1 = data_array1[1:, 1]
    y_values2 = data_array2[1:, 1]
    y_values3 = data_array3[1:, 1]
    y_values4 = data_array4[1:, 1]

    # Plot the data for each array
    plt.plot(x_values, y_values1, linestyle='-', color='b', label=y_axis1)
    plt.plot(x_values, y_values2, linestyle='-', color='r', label=y_axis2)
    # plt.plot(x_values, y_values3, linestyle='-', color='g', label=y_axis3)
    plt.plot(x_values, y_values4, linestyle='-', color='purple', label=y_axis4)

    # Add labels and a title
    plt.xlabel('$\textnormal{Time [sec]}$')
    plt.title(f"{y_axis1}, {y_axis2}, {y_axis3}, and {y_axis4} versus Time")

    # Add legends
    plt.legend()

    # Show the plot
    plt.show()

