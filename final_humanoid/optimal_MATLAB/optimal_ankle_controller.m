function u = optimal_ankle_controller(state, BRS_data)
    % Extract BRS data
    BRS = BRS_data.BRS;
    theta_grid = BRS_data.theta_grid;
    theta_dot_grid = BRS_data.theta_dot_grid;
    tau_grid = BRS_data.tau_grid;
    
    % Find the closest grid point
    [~, i] = min(abs(theta_grid - state(1)));
    [~, j] = min(abs(theta_dot_grid - state(2)));
    [~, k] = min(abs(tau_grid - state(3)));
    
    % Check if the state is in the stabilizable region
    if BRS(i,j,k)
        % Calculate value function gradient with respect to tau
        % (This is a simplified approximation)
        if k < length(tau_grid)
            V_next = BRS(i,j,k+1);
            V_curr = BRS(i,j,k);
            grad_V_tau = (V_next - V_curr)/(tau_grid(k+1) - tau_grid(k));
        else
            V_curr = BRS(i,j,k);
            V_prev = BRS(i,j,k-1);
            grad_V_tau = (V_curr - V_prev)/(tau_grid(k) - tau_grid(k-1));
        end
        
        % Determine optimal control based on gradient
        % From the paper: u = MRTD_df if grad_V < 0, else u = -MRTD_pf
        if grad_V_tau < 0
            u = BRS_data.params.MRTD_df;  % Maximum dorsiflexion RTD
        else
            u = -BRS_data.params.MRTD_pf; % Maximum plantarflexion RTD
        end
    else
        % Outside stabilizable region - use emergency control
        % (In reality, would take a step)
        u = 0;
        warning('State outside stabilizable region');
    end
end