clear all;
close all;
clc;

%% Configuration Parameters (from config.yaml)
% System parameters
M_total = 60;                   % Total mass (kg)
H_total = 1.59;                 % Total height (m)
timestep = 0.0005;              % Simulation timestep (s)
sim_duration = 5;               % Simulation duration (s)
gravity = 9.81;                 % Gravity (m/s²)
damping = 2.5;                  % Joint damping (from MuJoCo model)

% Controller parameters (from config.yaml human:lqr_params)
Q_angle = 2000;                 % Cost on angle error
Q_velocity = 400;               % Cost on velocity error
R_cost = 0.01;                  % Cost on control effort

% Torque limits (from config.yaml)
max_torque_df = 22;             % Maximum dorsiflexion torque (Nm)
max_torque_pf = -88;            % Maximum plantarflexion torque (Nm) - negative
mrtd_df = 148;                  % Maximum rate of torque development for dorsiflexion (Nm/s)
mrtd_pf = 389;                  % Maximum rate of torque development for plantarflexion (Nm/s)

% Initial conditions (from config.yaml)
ankle_position_initial = -0.01; % Initial ankle angle (rad)
ankle_velocity_initial = 0.0;   % Initial ankle velocity (rad/s)

% Target position (from config.yaml)
ankle_position_setpoint = 0.0;  % Target ankle position (rad)

% Gear ratio (from MuJoCo model)
gear_ratio = 15;                % Actuator gear ratio

%% Compute anthropometric parameters (from xml_utilities.py)
m_feet = 2 * 0.0145 * M_total;  % Mass of feet (kg)
m_body = M_total - m_feet;      % Mass of body (kg)
l_COM = 0.575 * H_total;        % Distance to COM from ankle (m)
I = m_body * l_COM^2;           % Moment of inertia (kg·m²)

% Print configuration for verification
fprintf('---- Simulation Configuration ----\n');
fprintf('Mass: %.2f kg, Height: %.2f m, COM height: %.2f m\n', M_total, H_total, l_COM);
fprintf('Q_angle: %.1f, Q_velocity: %.1f, R: %.5f\n', Q_angle, Q_velocity, R_cost);
fprintf('Max torque (DF/PF): %.1f/%.1f Nm\n', max_torque_df, max_torque_pf);
fprintf('Max RTD (DF/PF): %.1f/%.1f Nm/s\n', mrtd_df, mrtd_pf);
fprintf('-------------------------------\n\n');

%% LQR Controller Design (from HumanLQRController._compute_human_only_lqr_gains)
% System matrices for inverted pendulum
%A = [0, 1;
%     m_body*gravity*l_COM/I, -damping/I];

effective_COM_height = l_COM * cos(ankle_position_initial);
A = [0, 1;
     m_body*gravity*effective_COM_height/I, -damping/I];
B = [0;
     1/I];

% Cost matrices
Q = diag([Q_angle, Q_velocity]);
R = R_cost;

% Compute LQR gains
[K, S, P] = lqr(A, B, Q, R);
fprintf('LQR gain matrix K = [%.4f %.4f]\n\n', K(1), K(2));

%% Simulation Setup
% Time vector
t = 0:timestep:sim_duration;
num_steps = length(t);

% State vector: [ankle_angle; ankle_velocity]
x = zeros(2, num_steps);
x(:,1) = [ankle_position_initial; ankle_velocity_initial];

% Control and torque variables
u_raw = zeros(1, num_steps);          % Raw LQR control (before limits)
u_mt_limited = zeros(1, num_steps);   % After MT limits
u_final = zeros(1, num_steps);        % After MRTD limits (final applied torque)
rtd = zeros(1, num_steps);            % Rate of torque development
rtd_limit = zeros(1, num_steps);      % Applied RTD limit

%% Simulation Loop
for i = 1:num_steps-1
    % Current state
    current_state = x(:,i);
    
    % Reference state (target position with zero velocity)
    reference_state = [ankle_position_setpoint; 0];
    
    % Error state
    error_state = current_state - reference_state;
    
    % Compute raw LQR control
    u_raw(i) = -K * error_state;
    
    % Apply maximum torque limits (MT)
    max_df = max_torque_df / gear_ratio;
    max_pf = max_torque_pf / gear_ratio;
    u_mt_limited(i) = min(max(u_raw(i), max_pf), max_df);
    
    % Apply rate of torque development limits (MRTD)
    if i == 1
        % First step - no previous torque
        delta_torque = u_mt_limited(i);
        rtd(i) = delta_torque / timestep;
        prev_torque = 0;
    else
        delta_torque = u_mt_limited(i) - u_final(i-1);
        rtd(i) = delta_torque / timestep;
        prev_torque = u_final(i-1);
    end
    
    % Determine RTD limit based on direction
    if delta_torque > 0
        max_increase = (mrtd_df / gear_ratio) * timestep;
        rtd_limit(i) = mrtd_df;
        delta_torque = min(delta_torque, max_increase);
    else
        max_decrease = -(mrtd_pf / gear_ratio) * timestep;
        rtd_limit(i) = -mrtd_pf;
        delta_torque = max(delta_torque, max_decrease);
    end
    
    % Apply MRTD limit
    u_final(i) = prev_torque + delta_torque;
    
    % Additional safety check for MT limits
    u_final(i) = min(max(u_final(i), max_pf), max_df);
    
    % Calculate actual torque applied to ankle (scaled by gear ratio)
    tau_human = u_final(i) * gear_ratio;
    
    % System dynamics: x_dot = Ax + Bu
    x_dot = A * current_state + B * tau_human;
    
    % Euler integration
    x(:,i+1) = current_state + timestep * x_dot;
end

% For the last time step (only for plotting)
u_raw(end) = -K * (x(:,end) - reference_state);
u_mt_limited(end) = min(max(u_raw(end), max_pf), max_df);
u_final(end) = u_mt_limited(end); % Simplification for last point
rtd(end) = rtd(end-1);
rtd_limit(end) = rtd_limit(end-1);

% Calculate joint torques (scaled by gear ratio)
human_torque = u_final * gear_ratio;

%% Visualization
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Ankle Position
subplot(2, 2, 1);
plot(t, x(1,:)*180/pi, 'LineWidth', 2);
hold on;
plot(t, ones(size(t))*ankle_position_setpoint*180/pi, 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Ankle Angle (deg)');
title('Ankle Joint Position');
legend('Actual Position', 'Target Position');

% Plot 2: Ankle Velocity
subplot(2, 2, 2);
plot(t, x(2,:)*180/pi, 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (deg/s)');
title('Ankle Joint Velocity');

% Plot 3: Torque Components
subplot(2, 2, 3);
plot(t, u_raw*gear_ratio, 'b--', 'LineWidth', 1);
hold on;
plot(t, u_mt_limited*gear_ratio, 'g-.', 'LineWidth', 1);
plot(t, human_torque, 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Control Torques');
legend('Raw LQR Torque', 'After MT Limits', 'After MRTD Limits (Final)');
% Add lines for torque limits
hold on;
plot(t, ones(size(t))*max_torque_df, 'k--');
plot(t, ones(size(t))*max_torque_pf, 'k--');

% Plot 4: Rate of Torque Development
subplot(2, 2, 4);
plot(t, rtd*gear_ratio, 'LineWidth', 2);
hold on;
plot(t, ones(size(t))*mrtd_df, 'k--', 'LineWidth', 1);
plot(t, ones(size(t))*(-mrtd_pf), 'k--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('RTD (Nm/s)');
title('Rate of Torque Development');
legend('Actual RTD', 'RTD Limits');

% Adjust layout
sgtitle('Human Ankle LQR Control Simulation', 'FontSize', 16);
set(gcf, 'Color', 'w');

%% Display Summary Statistics
fprintf('---- Simulation Results ----\n');
fprintf('Final ankle position: %.4f deg\n', x(1,end)*180/pi);
fprintf('Position error: %.4f deg\n', (x(1,end) - ankle_position_setpoint)*180/pi);
fprintf('Max torque applied: %.2f Nm\n', max(abs(human_torque)));
fprintf('Max RTD: %.2f Nm/s\n', max(abs(rtd)*gear_ratio));
fprintf('--------------------------\n');

%% Optional: Phase Space Visualization
figure('Position', [500, 300, 800, 600]);
plot(x(1,:)*180/pi, x(2,:)*180/pi, 'b-', 'LineWidth', 2);
hold on;
plot(x(1,1)*180/pi, x(2,1)*180/pi, 'go', 'MarkerSize', 10, 'LineWidth', 2); % Start
plot(ankle_position_setpoint*180/pi, 0, 'r*', 'MarkerSize', 10, 'LineWidth', 2); % Target
grid on;
xlabel('Ankle Angle (deg)');
ylabel('Angular Velocity (deg/s)');
title('Phase Portrait');
legend('Trajectory', 'Initial State', 'Target State');
set(gcf, 'Color', 'w');

% Create a text table with key parameters
annotation('textbox', [0.02, 0.02, 0.4, 0.25], ...
    'String', {['Mass: ' num2str(M_total) ' kg'], ...
               ['COM height: ' num2str(l_COM) ' m'], ...
               ['Damping: ' num2str(damping) ' Nms/rad'], ...
               ['K = [' num2str(K(1), '%.2f') ' ' num2str(K(2), '%.2f') ']'], ...
               ['Max DF/PF torque: ' num2str(max_torque_df) '/' num2str(max_torque_pf) ' Nm'], ...
               ['Max DF/PF RTD: ' num2str(mrtd_df) '/' num2str(mrtd_pf) ' Nm/s']}, ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', [0.9, 0.9, 0.9], ...
    'EdgeColor', 'k');