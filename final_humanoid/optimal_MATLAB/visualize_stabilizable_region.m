function visualize_stabilizable_region(BRS_data)
    % Extract data
    BRS = BRS_data.BRS;
    theta_grid = BRS_data.theta_grid;
    theta_dot_grid = BRS_data.theta_dot_grid;
    tau_grid = BRS_data.tau_grid;
    
    % Create a 2D slice at a specific torque value
    tau_index = round(length(tau_grid)/2);  % Middle torque value
    SR_slice = squeeze(BRS(:,:,tau_index));
    
    % Create figure
    figure('Position', [200, 200, 800, 600]);
    
    % Plot stabilizable region
    imagesc(theta_grid*180/pi-90, theta_dot_grid*180/pi, SR_slice');
    set(gca, 'YDir', 'normal');
    colormap([1 1 1; 0 0.4 0.8]);  % White for outside SR, blue for inside
    hold on;
    
    % Create contour for better visualization
    contour(theta_grid*180/pi-90, theta_dot_grid*180/pi, double(SR_slice'), [0.5 0.5], 'r', 'LineWidth', 2);
    
    % Add labels
    xlabel('Ankle Angle (deg from vertical)');
    ylabel('Ankle Angular Velocity (deg/s)');
    title(['Stabilizable Region at τ = ' num2str(tau_grid(tau_index), '%.1f') ' Nm']);
    
    % Add XCoM boundary for comparison (from Hof model)
    % XCoM boundary is ω = sqrt(g/l)·x as per the paper
    l = BRS_data.params.l_COM;
    g = BRS_data.params.g;
    omega0 = sqrt(g/l);
    x = linspace(-20, 20, 100);
    plot(x, omega0*x*180/pi, 'k--', 'LineWidth', 1.5);
    legend('Stabilizable Region Boundary', 'XCoM Boundary');
    
    grid on;
end