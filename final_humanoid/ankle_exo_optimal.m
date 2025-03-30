%% System Parameters
% Anthropometric parameters (from the paper and ankle_lqr_simulation.m)
M_total = 60;                   % Total body mass (kg)
H_total = 1.59;                 % Total height (m)
m_feet = 2 * 0.0145 * M_total;  % Mass of feet (kg)
m_body = M_total - m_feet;      % Mass of body (kg)
l_COM = 0.575 * H_total;        % Distance to COM from ankle (m)
I = m_body * l_COM^2;           % Moment of inertia (kg·m²)
g = 9.81;                       % Gravity (m/s²)
b = 2.5;                        % Damping coefficient (Nm·s/rad)

% Control constraints
MT_df = 22;                     % Maximum dorsiflexion torque (Nm)
MT_pf = -88;                    % Maximum plantarflexion torque (Nm)
MRTD_df = 148;                  % Maximum rate of torque development - dorsiflexion (Nm/s)
MRTD_pf = 389;                  % Maximum rate of torque development - plantarflexion (Nm/s)

% Initial conditions
theta_0 = -0.03;                % Initial ankle angle (rad)
theta_dot_0 = 0.3;              % Initial ankle angular velocity (rad/s)

% Target position
theta_target = 0.0;             % Target ankle angle (rad)

% Simulation parameters
dt = 0.0005;                    % Time step (s)
T_sim = 5;                      % Simulation duration (s)
N = round(T_sim/dt) + 1;        % Number of time steps

% Foot geometry and friction parameters
l_heel = 0.05 * H_total;        % Distance from ankle to heel (m)
l_toe = 0.15 * H_total;         % Distance from ankle to toe (m)
mu = 0.8;                       % Friction coefficient

% Exoskeleton parameters
exo_control_type = 'gravity_comp';  % Options: 'PD', 'gravity_comp', 'none'
exo_torque_limit = 50;           % Maximum exoskeleton torque (Nm)
Kp_exo = m_body * g * l_COM;     % Proportional gain for PD controller
Kd_exo = 0.3 * sqrt(m_body * l_COM^2 * Kp_exo); % Derivative gain

% Exoskeleton torque function (PD controller or gravity compensation)
function tau = exoTorqueFun(theta_val, theta_dot_val, control_type, theta_target, ...
                           Kp, Kd, torque_limit, m, g, l)
    switch control_type
        case 'PD'
            % PD controller: tries to hold ankle at target position
            raw_torque = Kp * (theta_target - theta_val) - Kd * theta_dot_val;
            
        case 'gravity_comp'
            % Gravity compensation: counters gravitational torque
            raw_torque = m * g * l * sin(theta_val);
            
        case 'none'
            % No exoskeleton assistance
            raw_torque = 0;
            
        otherwise
            error('Unknown exoskeleton control type');
    end
    
    % Apply torque limits
    tau = min(max(raw_torque, -torque_limit), torque_limit);
end

% Create a handle to the exoskeleton torque function with fixed parameters
exoTorque = @(theta_val, theta_dot_val) exoTorqueFun(theta_val, theta_dot_val, ...
                                            exo_control_type, theta_target, ...
                                            Kp_exo, Kd_exo, exo_torque_limit, ...
                                            m_body, g, l_COM);

%% Setup for Direct Collocation Optimal Control
% Based on the "Discretized Optimal Trajectory" tutorial

% Create optimization variables
theta = optimvar('theta', N, 1);
theta_dot = optimvar('theta_dot', N, 1);
a = optimvar('a', N-1, 1, 'LowerBound', MT_pf, 'UpperBound', MT_df);
s = optimvar('s', N-1, 1, 'LowerBound', 0);

% Create optimization problem
ankleProb = optimproblem;

% Define objective function: minimize control effort (human torque only)
ankleProb.Objective = sum(s) * dt;

% Add constraint for the norm of acceleration (using s as helper variable)
% We can't use abs() directly with optimization variables
% Instead, we create two inequality constraints equivalent to |a(i)| <= s(i)
acons_pos = optimconstr(N-1);
acons_neg = optimconstr(N-1);
for i = 1:N-1
    acons_pos(i) = a(i) <= s(i);     % a(i) ≤ s(i)
    acons_neg(i) = -a(i) <= s(i);    % -a(i) ≤ s(i)
end

% Define equations of motion constraints
% θ̇(i) = θ̇(i-1) + dt*(τ/I + g)
v_dynamics = optimconstr(N-1);
for i = 2:N
    % Get exoskeleton torque expression
    exo_expr = fcn2optimexpr(exoTorque, theta(i-1), theta_dot(i-1));
    
    % Combined torque (human + exo)
    ankle_torque_expr = a(i-1) + exo_expr;
    
    % Dynamics with combined torque
    v_dynamics(i-1) = theta_dot(i) == theta_dot(i-1) + dt * ((ankle_torque_expr/I) - (b/I)*theta_dot(i-1) + (m_body*g*l_COM/I)*sin(theta(i-1)));
end

% θ(i) = θ(i-1) + dt*(θ̇(i-1) + θ̇(i))/2
p_dynamics = optimconstr(N-1);
for i = 2:N
    p_dynamics(i-1) = theta(i) == theta(i-1) + dt * (theta_dot(i-1) + theta_dot(i)) / 2;
end

% Rate of torque development constraints (for human torque only)
rtd_upper_constraints = optimconstr(N-2);
rtd_lower_constraints = optimconstr(N-2);
for i = 1:N-2
    rtd_upper_constraints(i) = (a(i+1) - a(i))/dt <= MRTD_df;  % Upper bound
    rtd_lower_constraints(i) = (a(i+1) - a(i))/dt >= -MRTD_pf; % Lower bound
end

% Initial conditions
initialcons = [theta(1) == theta_0; theta_dot(1) == theta_dot_0];

% Final conditions (want to reach target and stop)
finalcons = [theta(N) == theta_target; theta_dot(N) == 0];

% Foot-ground contact constraints
% Define state-dependent bound functions
% For tipping constraints:
h_upper_fun = @(theta_val) m_body * g * cos(theta_val) * l_toe;
h_lower_fun = @(theta_val) -m_body * g * cos(theta_val) * l_heel;
% For friction constraint:
friction_bound_fun = @(theta_val) mu * m_body * g * min(l_toe, l_heel);

% Create SEPARATE constraint arrays for upper and lower bounds
foot_upper_constraints = optimconstr(N-1);
foot_lower_constraints = optimconstr(N-1);

for i = 1:N-1
    % Get exoskeleton torque expression
    exo_expr = fcn2optimexpr(exoTorque, theta(i), theta_dot(i));
    
    % Net ankle torque (human + exo)
    ankle_torque = a(i) + exo_expr;
    
    % Using fcn2optimexpr to handle nonlinear expressions
    tau_upper_tipping = fcn2optimexpr(h_upper_fun, theta(i));
    tau_lower_tipping = fcn2optimexpr(h_lower_fun, theta(i));
    tau_friction_bound = fcn2optimexpr(friction_bound_fun, theta(i));
    
    % Apply most restrictive constraint (tipping vs. friction)
    % For friction, we need both upper and lower bounds
    tau_upper = fcn2optimexpr(@(u1, u2) min(u1, u2), tau_upper_tipping, tau_friction_bound);
    tau_lower = fcn2optimexpr(@(l1, l2) max(l1, l2), tau_lower_tipping, -tau_friction_bound);
    
    % Add constraints to the problem as SEPARATE constraints
    foot_upper_constraints(i) = ankle_torque <= tau_upper;
    foot_lower_constraints(i) = ankle_torque >= tau_lower;
end

% Add angle range constraint to prevent extreme angles
max_angle = 30 * pi/180;  % 30 degrees in radians
angle_range_upper = optimconstr(N);
angle_range_lower = optimconstr(N);
for i = 1:N
    angle_range_upper(i) = theta(i) <= max_angle;
    angle_range_lower(i) = theta(i) >= -max_angle;
end

% Add all constraints to the problem
ankleProb.Constraints.acons_pos = acons_pos;
ankleProb.Constraints.acons_neg = acons_neg;
ankleProb.Constraints.v_dynamics = v_dynamics;
ankleProb.Constraints.p_dynamics = p_dynamics;
ankleProb.Constraints.rtd_upper_constraints = rtd_upper_constraints;
ankleProb.Constraints.rtd_lower_constraints = rtd_lower_constraints;
ankleProb.Constraints.initialcons = initialcons;
ankleProb.Constraints.finalcons = finalcons;
ankleProb.Constraints.foot_upper_constraints = foot_upper_constraints;
ankleProb.Constraints.foot_lower_constraints = foot_lower_constraints;
ankleProb.Constraints.angle_range_upper = angle_range_upper;
ankleProb.Constraints.angle_range_lower = angle_range_lower;

% Create an initial guess for the variables
x0.theta = linspace(theta_0, theta_target, N)';
x0.theta_dot = linspace(theta_dot_0, 0, N)';
x0.a = zeros(N-1, 1);
x0.s = ones(N-1, 1);

% Set options for the solver
options = optimoptions('fmincon', 'Display', 'iter', ...
                      'Algorithm', 'interior-point', ...
                      'MaxFunctionEvaluations', 10000, ...
                      'MaxIterations', 1000, ...
                      'OptimalityTolerance', 1e-6);

% Solve the problem
[sol, fval, exitflag, output] = solve(ankleProb, x0, 'Options', options);

% Check if a solution was found
if exitflag > 0
    disp('Optimal solution found!');
else
    warning('Problem not solved successfully. Exitflag: %d', exitflag);
    disp(output.message);
end

%% Extract solution and propagate dynamics if necessary
if exitflag > 0
    % Extract the solution
    theta_sol = sol.theta;
    theta_dot_sol = sol.theta_dot;
    a_sol = sol.a;  % Human torque
    s_sol = sol.s;
    
    % Create time vector
    t = (0:N-1) * dt;
    
    % Calculate exoskeleton torque
    exo_torque_sol = zeros(N-1, 1);
    for i = 1:N-1
        exo_torque_sol(i) = exoTorque(theta_sol(i), theta_dot_sol(i));
    end
    
    % Calculate total ankle torque
    ankle_torque_sol = a_sol + exo_torque_sol;
else
    % If optimization fails, fallback to a simple control strategy
    disp('Using fallback control strategy...');
    
    % Allocate arrays
    theta_sol = zeros(N, 1);
    theta_dot_sol = zeros(N, 1);
    a_sol = zeros(N-1, 1);          % Human torque
    exo_torque_sol = zeros(N-1, 1); % Exo torque
    ankle_torque_sol = zeros(N-1, 1); % Combined torque
    
    % Set initial conditions
    theta_sol(1) = theta_0;
    theta_dot_sol(1) = theta_dot_0;
    
    % Simple PD control loop
    Kp = 200;
    Kd = 50;
    
    for i = 1:N-1
        % Calculate exoskeleton torque
        exo_torque_sol(i) = exoTorque(theta_sol(i), theta_dot_sol(i));
        
        % Calculate control with PD feedback
        error = theta_target - theta_sol(i);
        error_dot = -theta_dot_sol(i);
        
        % Human control (PD)
        a_sol(i) = Kp * error + Kd * error_dot;
        
        % Apply human torque limits
        a_sol(i) = min(max(a_sol(i), MT_pf), MT_df);
        
        % Combined torque
        ankle_torque_sol(i) = a_sol(i) + exo_torque_sol(i);
        
        % Apply foot-ground constraints
        % Calculate bounds for this state
        tau_upper_tipping = h_upper_fun(theta_sol(i));
        tau_lower_tipping = h_lower_fun(theta_sol(i));
        tau_friction_bound = friction_bound_fun(theta_sol(i));
        
        % Combined bounds
        tau_upper = min(tau_upper_tipping, tau_friction_bound);
        tau_lower = max(tau_lower_tipping, -tau_friction_bound);
        
        % Check if ankle torque exceeds bounds
        if ankle_torque_sol(i) > tau_upper
            % Reduce human torque to satisfy bound
            a_sol(i) = tau_upper - exo_torque_sol(i);
            ankle_torque_sol(i) = tau_upper;
        elseif ankle_torque_sol(i) < tau_lower
            % Increase human torque to satisfy bound
            a_sol(i) = tau_lower - exo_torque_sol(i);
            ankle_torque_sol(i) = tau_lower;
        end
        
        % Simple Euler integration using combined torque
        theta_dot_sol(i+1) = theta_dot_sol(i) + dt * ((ankle_torque_sol(i)/I) - (b/I)*theta_dot_sol(i) + (m_body*g*l_COM/I)*sin(theta_sol(i)));
        theta_sol(i+1) = theta_sol(i) + dt * theta_dot_sol(i);
    end
    
    % Create time vector
    t = (0:N-1) * dt;
end

%% Compute Derived Quantities
% Calculate torque rate of change for analysis
rtd_sol = zeros(N-1, 1);
for i = 1:N-2
    rtd_sol(i) = (a_sol(i+1) - a_sol(i)) / dt;
end
rtd_sol(N-1) = rtd_sol(N-2);  % Repeat the last value for plotting

% Calculate the gravity torque
gravity_torque = m_body * g * l_COM * sin(theta_sol);

% Calculate the foot-ground constraint bounds along the trajectory
tipping_upper_bound = zeros(N, 1);
tipping_lower_bound = zeros(N, 1);
friction_bound = zeros(N, 1);
for i = 1:N
    tipping_upper_bound(i) = h_upper_fun(theta_sol(i));
    tipping_lower_bound(i) = h_lower_fun(theta_sol(i));
    friction_bound(i) = friction_bound_fun(theta_sol(i));
end

% Calculate combined bounds
upper_bound = min(tipping_upper_bound, friction_bound);
lower_bound = max(tipping_lower_bound, -friction_bound);

%% Export torque trajectory to CSV
% Create a table with time and applied human torque
torque_data = [t'; [a_sol; a_sol(end)]'];

% Save to CSV file
csv_filename = 'optimal_ankle_torque_trajectory.csv';
writematrix(torque_data, csv_filename);
fprintf('Human torque trajectory saved to: %s\n', csv_filename);

% Also save exo torque for reference
exo_torque_data = [t(1:N-1)'; exo_torque_sol];
exo_csv_filename = 'exo_ankle_torque_trajectory.csv';
writematrix(exo_torque_data, exo_csv_filename);
fprintf('Exo torque trajectory saved to: %s\n', exo_csv_filename);

%% Visualization
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Ankle Position
subplot(2, 2, 1);
plot(t, theta_sol*180/pi, 'LineWidth', 2);
hold on;
plot(t, ones(size(t))*theta_target*180/pi, 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Ankle Angle (deg)');
title('Ankle Joint Position');
legend('Actual Position', 'Target Position');

% Plot 2: Ankle Velocity
subplot(2, 2, 2);
plot(t, theta_dot_sol*180/pi, 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (deg/s)');
title('Ankle Joint Velocity');

% Plot 3: Torque Components
subplot(2, 2, 3);
plot(t(1:N-1), a_sol, 'b-', 'LineWidth', 2);
hold on;
plot(t(1:N-1), exo_torque_sol, 'g-', 'LineWidth', 2);
plot(t(1:N-1), ankle_torque_sol, 'r-', 'LineWidth', 2);
plot(t, gravity_torque, 'm--', 'LineWidth', 1.5);
plot(t, upper_bound, 'k-.', 'LineWidth', 1);
plot(t, lower_bound, 'k-.', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Ankle Joint Torque Components');
legend('Human Torque', 'Exo Torque', 'Combined Torque', 'Gravity Torque', 'Foot-Ground Bounds');

% Plot 4: Rate of Torque Development (human only)
subplot(2, 2, 4);
plot(t(1:N-1), rtd_sol, 'LineWidth', 2);
hold on;
plot(t, ones(size(t))*MRTD_df, 'r--', 'LineWidth', 1);
plot(t, -ones(size(t))*MRTD_pf, 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('RTD (Nm/s)');
title('Human Rate of Torque Development');
legend('Applied RTD', 'RTD Limits');

% Adjust layout
sgtitle(['Optimal Ankle Balance Control with ' exo_control_type ' Exoskeleton Assistance'], 'FontSize', 16);
set(gcf, 'Color', 'w');

%% Additional Analysis: Phase Plane Trajectory
figure('Position', [100, 550, 600, 500]);
plot(theta_sol*180/pi, theta_dot_sol*180/pi, 'b-', 'LineWidth', 2);
hold on;
plot(theta_sol(1)*180/pi, theta_dot_sol(1)*180/pi, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
plot(theta_target*180/pi, 0, 'gx', 'MarkerSize', 8, 'LineWidth', 2);
grid on;
xlabel('Ankle Angle (deg)');
ylabel('Angular Velocity (deg/s)');
title('Phase Plane Trajectory');
legend('Trajectory', 'Initial State', 'Target State');

%% Additional Analysis: Exoskeleton Contribution
figure('Position', [700, 550, 700, 500]);
subplot(2,1,1);
plot(t(1:N-1), a_sol, 'b-', 'LineWidth', 2);
hold on;
plot(t(1:N-1), exo_torque_sol, 'g-', 'LineWidth', 2);
plot(t(1:N-1), ankle_torque_sol, 'r-', 'LineWidth', 2);
plot(t, gravity_torque(1:N), 'm--', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Torque Contribution Analysis');
legend('Human Torque', 'Exo Torque', 'Combined Torque', 'Gravity Torque');

subplot(2,1,2);
% Calculate percentage of exo contribution to total torque
exo_percentage = zeros(N-1, 1);
for i = 1:N-1
    if abs(ankle_torque_sol(i)) > 0.01  % Avoid division by near-zero
        exo_percentage(i) = 100 * exo_torque_sol(i) / ankle_torque_sol(i);
    else
        exo_percentage(i) = 0;
    end
end
plot(t(1:N-1), exo_percentage, 'g-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Exo Contribution (%)');
title('Exoskeleton Torque Contribution Percentage');
ylim([-100, 100]);

%% Display Summary Statistics
fprintf('\n---- Simulation Results ----\n');
fprintf('Final ankle position: %.4f deg\n', theta_sol(end)*180/pi);
fprintf('Position error: %.4f deg\n', (theta_sol(end) - theta_target)*180/pi);
fprintf('Maximum ankle angle: %.4f deg\n', max(abs(theta_sol))*180/pi);
fprintf('Maximum angular velocity: %.4f deg/s\n', max(abs(theta_dot_sol))*180/pi);
fprintf('Maximum human torque: %.4f Nm\n', max(abs(a_sol)));
fprintf('Maximum exo torque: %.4f Nm\n', max(abs(exo_torque_sol)));
fprintf('Maximum combined torque: %.4f Nm\n', max(abs(ankle_torque_sol)));
fprintf('Maximum human RTD: %.4f Nm/s\n', max(abs(rtd_sol)));
fprintf('--------------------------\n');

% Check if solution respects foot-ground constraints
torque_within_bounds = all(ankle_torque_sol <= upper_bound(1:N-1) & ankle_torque_sol >= lower_bound(1:N-1));
fprintf('Solution respects foot-ground constraints: %s\n', mat2str(torque_within_bounds));

% Check which constraint is more restrictive (tipping or friction)
tipping_more_restrictive_upper = sum(tipping_upper_bound < friction_bound);
tipping_more_restrictive_lower = sum(tipping_lower_bound > -friction_bound);
fprintf('Tipping constraint more restrictive than friction: %.1f%% (upper), %.1f%% (lower)\n', ...
    100*tipping_more_restrictive_upper/N, 100*tipping_more_restrictive_lower/N);

% Exoskeleton contribution analysis
average_exo_contribution = mean(abs(exo_torque_sol)) / mean(abs(ankle_torque_sol)) * 100;
fprintf('Average exoskeleton torque contribution: %.1f%%\n', average_exo_contribution);