function BRS = simple_backward_reach(params, target_func, max_time, dt)
    % A simplified backward reachability analysis
    
    % Create a grid in state space
    N_points = 50;  % Number of points in each dimension (for a simple example)
    
    % Define grid bounds
    theta_min = pi/2 - 0.3;  
    theta_max = pi/2 + 0.3;
    theta_dot_min = -0.5;
    theta_dot_max = 0.5;
    tau_min = params.MT_pf;
    tau_max = params.MT_df;
    
    % Create grid arrays
    theta_grid = linspace(theta_min, theta_max, N_points);
    theta_dot_grid = linspace(theta_dot_min, theta_dot_max, N_points);
    tau_grid = linspace(tau_min, tau_max, N_points);
    
    % Initialize the BRS with just the target set
    BRS = false(N_points, N_points, N_points);
    
    % First, mark all states that are directly in the target set
    for i = 1:N_points
        for j = 1:N_points
            for k = 1:N_points
                state = [theta_grid(i); theta_dot_grid(j); tau_grid(k)];
                if target_func(state)
                    BRS(i,j,k) = true;
                end
            end
        end
    end
    
    % Backward reachability iteration
    current_time = 0;
    while current_time < max_time
        % Make a copy of the current BRS
        new_BRS = BRS;
        
        % Check each grid point
        for i = 1:N_points
            for j = 1:N_points
                for k = 1:N_points
                    % Skip points already in the BRS
                    if BRS(i,j,k)
                        continue;
                    end
                    
                    % Current state
                    state = [theta_grid(i); theta_dot_grid(j); tau_grid(k)];
                    
                    % Try both control extremes
                    reached_target = false;
                    
                    % Try maximum dorsiflexion RTD
                    u = params.MRTD_df;
                    dx = ankle_dynamics(state, u, params);
                    next_state = state + dx*dt;
                    
                    % Find the closest grid point
                    [~, ni] = min(abs(theta_grid - next_state(1)));
                    [~, nj] = min(abs(theta_dot_grid - next_state(2)));
                    [~, nk] = min(abs(tau_grid - next_state(3)));
                    
                    % Check if next state is in BRS
                    if BRS(ni,nj,nk)
                        reached_target = true;
                    end
                    
                    % If not, try minimum plantarflexion RTD
                    if ~reached_target
                        u = -params.MRTD_pf;
                        dx = ankle_dynamics(state, u, params);
                        next_state = state + dx*dt;
                        
                        % Find the closest grid point
                        [~, ni] = min(abs(theta_grid - next_state(1)));
                        [~, nj] = min(abs(theta_dot_grid - next_state(2)));
                        [~, nk] = min(abs(tau_grid - next_state(3)));
                        
                        % Check if next state is in BRS
                        if BRS(ni,nj,nk)
                            reached_target = true;
                        end
                    end
                    
                    % Add state to BRS if it can reach the target
                    if reached_target
                        new_BRS(i,j,k) = true;
                    end
                end
            end
        end
        
        % Update BRS
        BRS = new_BRS;
        current_time = current_time + dt;
        
        % Display progress
        disp(['Backward reachability: ' num2str(current_time) ' of ' num2str(max_time) ' seconds']);
    end
    
    % Return grid information along with BRS
    BRS_data.BRS = BRS;
    BRS_data.theta_grid = theta_grid;
    BRS_data.theta_dot_grid = theta_dot_grid;
    BRS_data.tau_grid = tau_grid;
    
    BRS = BRS_data;
end