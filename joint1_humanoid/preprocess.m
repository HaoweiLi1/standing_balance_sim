function preprocess_traj(input_traj_file, output_traj_file)
    % PREPROCESS_TRAJ - Preprocess trajectory data to match MuJoCo timestep
    %
    % This function reads a trajectory from a MATLAB simulation output,
    % reads the MuJoCo timestep from config.yaml, and interpolates the
    % trajectory to match the MuJoCo timestep.
    %
    % Syntax:
    %   preprocess_traj(input_traj_file, output_traj_file)
    %
    % Inputs:
    %   input_traj_file - Path to the input trajectory CSV file
    %   output_traj_file - Path to save the processed trajectory CSV file
    %
    % Example:
    %   preprocess_traj('matlab_trajectory.csv', 'ankle_torque_trajectory.csv')
    %
    % Notes:
    %   - The input CSV should have two columns: time and torque
    %   - The config.yaml file must be in the current directory
    %   - The first column of the input CSV is assumed to be time in seconds
    
    % Check input arguments
    if nargin < 2
        error('Not enough input arguments. Usage: preprocess_traj(input_traj_file, output_traj_file)');
    end
    
    % Read the input trajectory
    try
        data = readmatrix(input_traj_file);
        fprintf('Successfully loaded input trajectory from: %s\n', input_traj_file);
    catch ME
        error('Failed to read input file: %s\nError: %s', input_traj_file, ME.message);
    end
    
    % Check data format
    if size(data, 2) < 2
        error('Input CSV must have at least 2 columns (time, torque)');
    end
    
    % Extract time and torque data
    time_original = data(:, 1);
    torque_original = data(:, 2);
    
    % Compute the original timestep
    original_timesteps = diff(time_original);
    original_dt = mean(original_timesteps);
    original_dt_std = std(original_timesteps);
    
    % Check if the original timesteps are consistent
    if original_dt_std > 1e-6
        warning(['Original timesteps are not uniform. Mean: ' num2str(original_dt) ...
                 ', Std: ' num2str(original_dt_std)]);
    end
    
    fprintf('Original trajectory timestep: %.6f seconds\n', original_dt);
    
    % Read MuJoCo timestep from config.yaml
    mujoco_dt = read_mujoco_timestep('config.yaml');
    fprintf('MuJoCo simulation timestep: %.6f seconds\n', mujoco_dt);
    
    % Verify timestep relationship
    timestep_ratio = original_dt / mujoco_dt;
    is_integer_multiple = abs(round(timestep_ratio) - timestep_ratio) < 1e-6;
    
    if ~is_integer_multiple
        % If not an integer multiple, warn the user
        warning(['The original timestep (%.6f) is not an integer multiple ' ...
                 'of the MuJoCo timestep (%.6f). Ratio: %.4f\n' ...
                 'This may affect control accuracy.'], ...
                original_dt, mujoco_dt, timestep_ratio);
    else
        fprintf('Original timestep is %.0f times the MuJoCo timestep\n', round(timestep_ratio));
    end
    
    % Create a new time vector at MuJoCo timestep
    start_time = time_original(1);
    end_time = time_original(end);
    time_interp = start_time:mujoco_dt:end_time;
    
    % Choose interpolation method based on the data characteristics
    % Check for rapidly changing signals (high frequency components)
    fft_torque = fft(torque_original);
    power_spectrum = abs(fft_torque).^2;
    half_point = ceil(length(power_spectrum)/2);
    power_spectrum = power_spectrum(1:half_point);
    total_power = sum(power_spectrum);
    high_freq_power = sum(power_spectrum(ceil(half_point/2):end));
    high_freq_ratio = high_freq_power / total_power;
    
    % Adaptively select interpolation method
    if high_freq_ratio > 0.3
        % For signals with significant high-frequency components
        method = 'pchip';  % Piecewise Cubic Hermite Interpolating Polynomial
        fprintf('Using PCHIP interpolation for high-frequency content\n');
    elseif high_freq_ratio > 0.1
        % For moderate frequency content
        method = 'spline';
        fprintf('Using Spline interpolation for moderate-frequency content\n');
    else
        % For smoother, low-frequency signals
        method = 'linear';
        fprintf('Using Linear interpolation for low-frequency content\n');
    end
    
    % Perform interpolation
    torque_interp = interp1(time_original, torque_original, time_interp, method);
    
    % Create the output data
    output_data = [time_interp', torque_interp'];
    
    % Save the interpolated trajectory
    try
        writematrix(output_data, output_traj_file, 'Delimiter', ',');
        fprintf('Successfully wrote interpolated trajectory to: %s\n', output_traj_file);
    catch ME
        error('Failed to write output file: %s\nError: %s', output_traj_file, ME.message);
    end
    
    % Display summary statistics
    fprintf('\nInterpolation Summary:\n');
    fprintf('Original data: %d points, %.6f to %.6f seconds (dt=%.6f)\n', ...
        length(time_original), time_original(1), time_original(end), original_dt);
    fprintf('Interpolated data: %d points, %.6f to %.6f seconds (dt=%.6f)\n', ...
        length(time_interp), time_interp(1), time_interp(end), mujoco_dt);
    fprintf('Interpolation method: %s\n', method);
    fprintf('File size increased by factor: %.1f\n', length(time_interp)/length(time_original));
    
    % Optional: Plot for verification
    figure;
    subplot(2,1,1);
    plot(time_original, torque_original, 'b.-', 'DisplayName', 'Original');
    hold on;
    plot(time_interp, torque_interp, 'r-', 'DisplayName', 'Interpolated');
    xlabel('Time (s)');
    ylabel('Torque (Nm)');
    title('Trajectory Comparison');
    legend('show');
    grid on;
    
    % Zoom in to show interpolation details
    subplot(2,1,2);
    % Select a small region (5% of the total time) to zoom in
    zoom_center = (time_original(1) + time_original(end))/2;
    zoom_width = (time_original(end) - time_original(1))*0.05;
    zoom_start = zoom_center - zoom_width/2;
    zoom_end = zoom_center + zoom_width/2;
    
    plot(time_original, torque_original, 'bo', 'DisplayName', 'Original');
    hold on;
    plot(time_interp, torque_interp, 'r.', 'DisplayName', 'Interpolated');
    xlabel('Time (s)');
    ylabel('Torque (Nm)');
    title('Zoom View of Interpolation');
    xlim([zoom_start, zoom_end]);
    grid on;
    legend('show');
    
    end
    
    function dt = read_mujoco_timestep(config_file)
    % READ_MUJOCO_TIMESTEP - Read the simulation timestep from config.yaml
    %
    % This function reads the simulation_timestep parameter from the config.yaml
    % file used by the MuJoCo simulation.
    
        % Check if file exists
        if ~exist(config_file, 'file')
            error('Config file not found: %s', config_file);
        end
        
        % Read the file line by line
        fid = fopen(config_file, 'r');
        if fid == -1
            error('Failed to open config file: %s', config_file);
        end
        
        dt = -1;  % Default value if not found
        
        % Parse the file looking for simulation_timestep
        try
            line = fgetl(fid);
            while ischar(line)
                % Look for the simulation_timestep parameter
                if contains(line, 'simulation_timestep:')
                    % Extract the value using regular expressions
                    pattern = 'simulation_timestep:\s*([0-9.]+)';
                    tokens = regexp(line, pattern, 'tokens');
                    
                    if ~isempty(tokens)
                        dt = str2double(tokens{1}{1});
                        break;
                    end
                end
                line = fgetl(fid);
            end
        catch ME
            fclose(fid);
            error('Error parsing config file: %s', ME.message);
        end
        
        % Close the file
        fclose(fid);
        
        % Check if we found the timestep
        if dt == -1
            error('Could not find simulation_timestep in config file');
        end
    end