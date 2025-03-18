% System parameters from the paper (Older Female model)
M_total = 60;                   % Total mass (kg)
H_total = 1.59;                 % Height (m)
m_feet = 2*0.0145 * M_total;    % Mass of feet (kg)
m_body = M_total - m_feet;      % Mass of body (kg)
l_COM = 0.575 * H_total;        % Distance to COM (m)
I = m_body * l_COM^2;           % Moment of inertia
g = 9.81;                       % Gravity (m/s²)
b = 2.5;                        % Damping coefficient

% Human ankle torque limits
MT_df = 22;                     % Maximum dorsiflexion torque (Nm)
MT_pf = -88;                    % Maximum plantarflexion torque (Nm)
MRTD_df = 148;                  % Maximum RTD for dorsiflexion (Nm/s)
MRTD_pf = 389;                  % Maximum RTD for plantarflexion (Nm/s)

% Foot geometry
l_foot = 0.152 * H_total;       % Foot length
h_f = 0.039 * H_total;          % Foot height
a = 0.19 * l_foot;              % Distance from heel to ankle

% Simulation parameters
dt = 0.0005;                    % Time step (s)
T = 5;                          % Simulation duration (s)

% Ankle equilibrium (upright standing)
theta_eq = pi/2;                    % Upright posture (rad)
theta_dot_eq = 0;                   % Zero velocity (rad/s)
tau_eq = m_body*g*l_COM;            % Gravity compensation torque (Nm)

% Create a simple spherical target set
target_radius_theta = 0.01;         % Small angle deviation (rad)
target_radius_theta_dot = 0.05;     % Small velocity (rad/s)
target_radius_tau = 5;              % Torque deviation (Nm)

% Function to check if a state is in the target set
is_in_target = @(x) ((x(1)-theta_eq)^2/target_radius_theta^2 + ...
                      (x(2)-theta_dot_eq)^2/target_radius_theta_dot^2 + ...
                      (x(3)-tau_eq)^2/target_radius_tau^2) <= 1;

% Package parameters
params.m_body = m_body;
params.l_COM = l_COM;
params.g = g;
params.b = b;
params.I = I;
params.MT_df = MT_df;
params.MT_pf = MT_pf;
params.MRTD_df = MRTD_df;
params.MRTD_pf = MRTD_pf;
params.l_foot = l_foot;
params.h_f = h_f;
params.a = a;

% Compute backward reachable set (this will take some time)
disp('Computing backward reachable set...');
BRS_data = simple_backward_reach(params, is_in_target, 1.0, 0.01);
BRS_data.params = params;
disp('Backward reachable set computed.');

% Setup simulation
N_steps = round(T/dt);
x = zeros(3, N_steps+1);

% Initial conditions
x(:,1) = [pi/2 - 0.1; 0.2; 0];  % Initial state: leaning back slightly with some velocity

% Control inputs
u = zeros(1, N_steps);

% Simulate
disp('Running simulation...');
for k = 1:N_steps
    % Get current state
    state = x(:,k);
    
    % Compute optimal control
    u(k) = optimal_ankle_controller(state, BRS_data);
    
    % Compute state derivatives
    dx = ankle_dynamics(state, u(k), params);
    
    % Update state using Euler integration
    x(:,k+1) = state + dx*dt;
    
    % Apply torque limits
    x(3,k+1) = min(max(x(3,k+1), MT_pf), MT_df);
    
    % Check foot-ground contact constraints
    theta_heel = acos(a/l_COM);
    theta_toe = acos((l_foot-a)/l_COM);
    
    if x(1,k+1) < theta_heel || x(1,k+1) > theta_toe
        disp(['Foot constraint violation at t = ' num2str(k*dt) 's']);
        break;
    end
    
    % Display progress periodically
    if mod(k, round(N_steps/10)) == 0
        disp(['Simulation progress: ' num2str(100*k/N_steps) '%']);
    end
end
disp('Simulation complete.');

% Time vector
t = 0:dt:T;
t = t(1:length(x(1,:)));  % Trim if simulation ended early

% Create figure
figure('Position', [100, 100, 1200, 800]);

% Plot ankle angle (relative to vertical)
subplot(4,1,1);
plot(t, (x(1,:)-pi/2)*180/pi, 'LineWidth', 2);  % Convert to degrees from vertical
hold on;
yline(0, 'r--');  % Reference line for vertical
xlabel('Time (s)');
ylabel('Angle (deg)');
title('Ankle Angle (Relative to Vertical)');
grid on;

% Plot ankle velocity
subplot(4,1,2);
plot(t, x(2,:)*180/pi, 'LineWidth', 2);  % Convert to deg/s
hold on;
yline(0, 'r--');  % Reference line for zero velocity
xlabel('Time (s)');
ylabel('Angular Velocity (deg/s)');
title('Ankle Angular Velocity');
grid on;

% Plot ankle torque
subplot(4,1,3);
plot(t, x(3,:), 'LineWidth', 2);
hold on;
yline(MT_df, 'r--');  % Upper torque limit
yline(MT_pf, 'r--');  % Lower torque limit
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Ankle Torque');
grid on;

% Plot control input (RTD)
subplot(4,1,4);
stairs(t(1:end-1), u, 'LineWidth', 2);
hold on;
yline(MRTD_df, 'r--');  % Upper RTD limit
yline(-MRTD_pf, 'r--');  % Lower RTD limit
xlabel('Time (s)');
ylabel('Rate of Torque Development (Nm/s)');
title('Control Input (RTD)');
grid on;

% Adjust layout
sgtitle('Human Ankle Optimal Control Simulation');