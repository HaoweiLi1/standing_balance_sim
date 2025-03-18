function dx = ankle_dynamics(x, u, params)
    % State variables
    theta = x(1);        % Ankle angle
    theta_dot = x(2);    % Ankle angular velocity
    tau = x(3);          % Human torque
    
    % Unpack parameters
    m = params.m_body;
    l = params.l_COM;
    g = params.g;
    b = params.b;
    I = params.I;
    
    % Equation 3 from the paper
    dtheta = theta_dot;
    dtheta_dot = -g/l*cos(theta) - b/I*theta_dot + 1/I*tau;
    dtau = u;            % Rate of torque development is the control input
    
    dx = [dtheta; dtheta_dot; dtau];
end
