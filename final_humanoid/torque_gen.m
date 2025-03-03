clear all; close all; clc;

%% Parameters
T = 1.5;                   % Simulation time (s)
N = 100;                   % Number of points
time = linspace(0, T, N)'; % Time vector
constant_torque = -7;      % Constant torque value (Nm)

%% Generate constant torque
torque = constant_torque * ones(size(time));

%% Create and save data
data = [time, torque];
csvwrite('optimal_human_torque.csv', data);
fprintf('Constant torque saved to constant_human_torque.csv\n');

%% Plot the torque profile
figure;
plot(time, torque, 'LineWidth', 2);
title('Constant Human Torque Profile');
xlabel('Time (s)');
ylabel('Torque (Nm)');
grid on;
ylim([0, 20]);

fprintf('Generated constant torque profile with value: %.2f Nm\n', constant_torque);
fprintf('Time range: 0 to %.2f seconds\n', T);
fprintf('Number of data points: %d\n', N);