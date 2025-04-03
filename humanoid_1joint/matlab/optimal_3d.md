# 3D Lifted System for Optimal Ankle Control

## 1. Theoretical Basis: The "3D Lifted System"

Unlike simpler problems where torque is directly the control, `ankle_3d_optimal.m` treats torque $\tau$ itself as part of the system state. Here's the outline:

### State and Control Variables

**States** at each discrete time step $i$:

- $\theta(i)$ – ankle angle.
- $\theta\_dot(i)$ – ankle angular velocity.
- $\tau(i)$ – ankle torque (lifted into the state).

**Control** at each step $i$:

- $u(i)$ – the rate of change of torque, sometimes called $\dot{\tau}$.

This arrangement is sometimes called a 3D system: $(\theta,\theta\_dot,\tau)$ form the three "dimensions" of the state.

### Objective Function

The script penalizes the sum of squared torque-rate values:

$$\min \sum_{i=1}^{N-1} u(i)^2 \Delta t.$$

By penalizing $u^2$, the solver promotes smooth and gradual torque changes rather than abrupt spikes.

### Dynamics Constraints

Three sets of updates are enforced from step $i-1$ to step $i$:

**Angle** (trapezoidal rule for $\theta$):

$$\theta(i) = \theta(i-1) + \Delta t \frac{\theta\_dot(i-1) + \theta\_dot(i)}{2}.$$

**Angular Velocity**:

$$\theta\_dot(i) = \theta\_dot(i-1) + \Delta t \left[\frac{\tau(i-1)}{I} - \frac{b}{I}\theta\_dot(i-1) + \frac{mgl\_COM}{I}\sin(\theta(i-1))\right].$$

**Torque as a state** (Euler update):

$$\tau(i) = \tau(i-1) + \Delta t \cdot u(i-1).$$

### Torque and Torque-Rate Bounds

- Torque $\tau$ is directly bounded $\tau \in [MT\_pf, MT\_df]$.
- The control $u$ (torque-rate) is bounded by $\pm MRTD$.

### Initial and Final Conditions

- At $i=1$: $\theta(1) = \theta_0, \theta\_dot(1) = \theta\_dot_0, \tau(1) = 0$.
- At $i=N$: $\theta(N) = \theta_{target}, \theta\_dot(N) = 0$, while $\tau(N)$ is unconstrained (can end at any feasible value).

## 2. Explanation of Important Code Segments

### A. Parameter Definitions

```matlab
% System mass, geometry, inertia, damping
M_total = 60; H_total = 1.59;
m_feet = 2*0.0145*M_total; 
m_body = M_total - m_feet;
l_COM = 0.575*H_total;
I = m_body*l_COM^2;
g = 9.81; b = 2.5;

% Torque and torque-rate constraints
MT_df = 22;   % Max dorsiflexion torque
MT_pf = -88;  % Max plantarflexion torque
MRTD_df = 148;  % Max torque rate dev (df)
MRTD_pf = 389;  % Max torque rate dev (pf)

% Initial angles, velocity
theta_0 = -0.03;
theta_dot_0 = 0.3;
theta_target = 0.0;

% Simulation time, step, # of steps
dt = 0.01;
T_sim = 5;
N = round(T_sim/dt) + 1;
```

As in many torque-based problems, sets mass/inertia, damping, gravity, plus the maximum feasible torque and rate-of-change.

Time is discretized to 5 seconds with a 0.01 step.

### B. Declaring Optimization Variables

```matlab
% The "3D lifted system" approach
theta = optimvar('theta', N, 1);
theta_dot = optimvar('theta_dot', N, 1);

% tau is now a state, with explicit torque bounds
tau = optimvar('tau', N, 1, 'LowerBound', MT_pf, 'UpperBound', MT_df);

% u is torque-rate, with bounds on how quickly torque can change
u = optimvar('u', N-1, 1, 'LowerBound', -MRTD_pf, 'UpperBound', MRTD_df);
```

Here, `tau` is no longer a direct "input" but a state variable.

`u` is the new control (torque rate).

### C. Objective Function: Minimizing $\sum u(i)^2$

```matlab
ankleProb = optimproblem;
ankleProb.Objective = sum(u.^2)*dt;
```

Summation of squared torque-rate multiplied by the time step $\Delta t$. Encourages smaller (smoother) torque rate changes.

### D. Discretized Dynamics Constraints

**Position (Angle) Update:**

```matlab
p_dynamics = optimconstr(N-1);
for i = 2:N
    p_dynamics(i-1) = theta(i) == theta(i-1) + dt * (theta_dot(i-1) + theta_dot(i))/2;
end
```

Uses the trapezoidal rule to link $\theta(i)$ and $\theta\_dot$.

**Velocity Update:**

```matlab
v_dynamics = optimconstr(N-1);
for i = 2:N
    v_dynamics(i-1) = ...
        theta_dot(i) == theta_dot(i-1) + dt * ...
            ((tau(i-1)/I) - (b/I)*theta_dot(i-1) + (m_body*g*l_COM/I)*sin(theta(i-1)));
end
```

Approximates $\dot{\theta}$ using the torque from the previous step, damping, and gravity.

**Torque Update:**

```matlab
tau_dynamics = optimconstr(N-1);
for i = 2:N
    tau_dynamics(i-1) = tau(i) == tau(i-1) + dt * u(i-1);
end
```

This is the "lifted" part: each next-step torque is old torque plus $\Delta t$ times the torque rate $u$.

### E. Initial and Final Conditions

```matlab
% Start: angle=theta_0, velocity=theta_dot_0, torque=0
initialcons = [theta(1) == theta_0; 
               theta_dot(1) == theta_dot_0;
               tau(1) == 0];

% End: angle=theta_target, velocity=0, torque free
finalcons = [theta(N) == theta_target; 
             theta_dot(N) == 0];
```

Contrasts with standard approaches, which don't necessarily treat torque as a state. Here it's set to zero initially. End torque is unconstrained.

### F. Angle Range Constraints

```matlab
max_angle = 30 * pi/180;
for i = 1:N
    angle_range_upper(i) = theta(i) <= max_angle;
    angle_range_lower(i) = theta(i) >= -max_angle;
end
```

Restricts the ankle angle to ±30°.

### G. Collecting Constraints

```matlab
ankleProb.Constraints.v_dynamics = v_dynamics;
ankleProb.Constraints.p_dynamics = p_dynamics;
ankleProb.Constraints.tau_dynamics = tau_dynamics;
ankleProb.Constraints.initialcons = initialcons;
ankleProb.Constraints.finalcons = finalcons;
ankleProb.Constraints.angle_range_upper = angle_range_upper;
ankleProb.Constraints.angle_range_lower = angle_range_lower;
```

Brings them all together so `solve` can process them.

### H. Solving

```matlab
% Provide an initial guess for the solver
x0.theta = linspace(theta_0, theta_target, N)';
x0.theta_dot = linspace(theta_dot_0, 0, N)';
x0.tau = zeros(N,1);
x0.u = zeros(N-1,1);

options = optimoptions('fmincon','Display','iter',...
    'Algorithm','interior-point',...);
[sol, fval, exitflag, output] = solve(ankleProb, x0,...
    'Solver','fmincon','Options',options);

% If successful, extract the solution, else fallback
```

With a good initial guess, the solver can converge more reliably.

The final results are stored in `sol.theta`, `sol.theta_dot`, `sol.tau`, and `sol.u`.

### I. Post-processing and Plots

The script then:

- Displays messages if `exitflag>0` or does a fallback PD approach otherwise.
- Plots $\theta$ vs. time, $\dot{\theta}$ vs. time, $\tau$ vs. time, and $u$ (torque rate) vs. time.
- Exports results to CSV, logs summary statistics (e.g., final angle, max torque).

## 3. Key Differences Compared to a Direct Torque Control Approach

- By including $\tau$ as a state, `ankle_3d_optimal.m` has a straightforward handle on torque and its derivative.
- The objective $\sum u^2$ punishes rapid torque changes, rather than punishing torque magnitude itself (except for the bounding constraints on $\tau$).
- Rate-of-torque constraints become trivial: $|u(i)| \leq ...$
- The solution can more smoothly shape the torque profile to meet final angle/velocity conditions without large RTD spikes.

## Summary

1. **"3D Lifted System"**: The script sets up $\theta, \theta\_dot$, and $\tau$ as states, with the new control $u = \dot{\tau}$.

2. **Objective**: Minimizes $\sum (u(i))^2$, ensuring torque changes are as gentle as possible.

3. **Constraints**:
   - Standard direct collocation constraints for $\theta, \theta\_dot$.
   - Euler integration for $\tau$.
   - Bounds on $\tau$ and $\dot{\tau}$.
   - Angle limits and final conditions.

4. **MATLAB Tools**: Uses `optimproblem`, `optimvar`, and `solve` from the "Discretized Optimal Trajectory" and "Nonlinear Constraints" tutorials, demonstrating a more advanced approach to controlling torque.
