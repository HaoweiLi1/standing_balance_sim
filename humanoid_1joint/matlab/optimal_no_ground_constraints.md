# Direct Collocation for Optimal Ankle Control

## 1. Theoretical Basis for the Implementation

The `ankle_optimal_no_foot_ground.m` implements a direct collocation approach to optimal control:

### State and Control Variables

**State:**
- $\theta(i)$: the ankle angle at step $i$.
- $\theta\_dot(i)$: the ankle angular velocity at step $i$.

**Control:**
- $a(i)$: the torque (or net "acceleration-like" input) at each discrete step.

Additionally, a "slack" variable $s(i)$ is used to capture the absolute value of $a(i)$ in the objective function.

### Objective Function

The code minimizes:

$$\sum_i s(i) \Delta t,$$

where $|a(i)| \leq s(i)$. By keeping $s(i)$ small, the solver is encouraged to keep the magnitude of torque $a(i)$ small. This is effectively an $L_1$-type penalty on torque amplitude.

### Dynamics and Constraints

**Discretized EoM:** Uses the trapezoidal rule for the ankle angle $\theta$. For velocity, it applies:

$$\theta\_dot(i) = \theta\_dot(i-1) + \Delta t \left[ \frac{a(i-1)}{I} - \frac{b}{I}\theta\_dot(i-1) + \frac{m \cdot g \cdot l\_COM}{I}\sin(\theta(i-1)) \right].$$

This enforces the relationship $\ddot{\theta} = \frac{1}{I}[a - b \cdot \dot{\theta} + m \cdot g \cdot l\_COM \cdot \sin(\theta)]$ at each time step.

**Torque and Rate-of-Torque-Development Limits:**

- $LowerBound \leq a(i) \leq UpperBound$, reflecting plantarflexion/dorsiflexion maximums.
- A difference constraint $\frac{a(i+1) - a(i)}{\Delta t} \leq MRTD\_df, \geq -MRTD\_pf$ ensures the torque doesn't ramp up or down too quickly.

**Foot-ground constraints:** In principle, foot-ground friction/tipping constraints were coded but are commented out here, hence "no_foot_ground."

**Boundary Conditions:**
- $\theta(1) = \theta_0, \theta\_dot(1) = \theta\_dot(0)$
- $\theta(N) = \theta\_target, \theta\_dot(N) = 0$

### Why Slack Variable $s(i)$?

Normally, one cannot directly write $\min \sum |a(i)|$ because absolute values are non-differentiable. The typical trick is to add a slack $s(i)$ and constraints $-s(i) \leq a(i) \leq s(i)$. Minimizing $\sum s(i)$ then encourages $|a(i)|$ to be as small as possible.

## 2. Explanation of Important Code Segments

Below is a step-by-step outline of the important sections in `ankle_optimal_no_foot_ground.m`. (Not every line is repeated, but the critical parts are highlighted.)

### A. System and Control Parameter Setup

```matlab
% Anthropometric parameters
M_total = 60;                   % kg
H_total = 1.59;                 % m
m_feet = 2 * 0.0145 * M_total;  % mass of feet
m_body = M_total - m_feet;      
l_COM = 0.575 * H_total;
I = m_body * l_COM^2; 
g = 9.81;
b = 2.5;                        % damping

% Control constraints
MT_df = 22;   % Max dorsiflexion torque (Nm)
MT_pf = -88;  % Max plantarflexion torque (Nm)
MRTD_df = 148;   % Max rate of torque dev. (df) in Nm/s
MRTD_pf = 389;   % Max rate of torque dev. (pf) in Nm/s
```

These lines define the anthropometric and physical parameters used in the dynamic equations and constraints.

MT_df and MT_pf provide torque bounds; MRTD_df/MRTD_pf define how fast torque can change.

### B. Discretized Time Horizon

```matlab
dt = 0.0005;   
T_sim = 0.5;
N = round(T_sim/dt) + 1;
```

Sets up the time step (0.0005 s) and total simulation horizon (0.5 s).

N is the number of discrete steps.

### C. Creating Optimization Variables

```matlab
theta = optimvar('theta', N, 1);
theta_dot = optimvar('theta_dot', N, 1);
a = optimvar('a', N-1, 1, 'LowerBound', MT_pf, 'UpperBound', MT_df);
s = optimvar('s', N-1, 1, 'LowerBound', 0);
```

- `theta(i)`, `theta_dot(i)`: angle and angular velocity at each time step.
- `a(i)`: torque at each time step (bounded by [MT_pf, MT_df]).
- `s(i)`: the slack variable for $|a(i)| \leq s(i)$.

### D. Objective Function and Slack Constraints

```matlab
ankleProb = optimproblem;

% The objective: minimize sum(s)*dt
ankleProb.Objective = sum(s)*dt;

% Constraints to enforce |a(i)| <= s(i):
acons_pos = optimconstr(N-1);
acons_neg = optimconstr(N-1);
for i = 1:N-1
    acons_pos(i) = a(i) <= s(i);
    acons_neg(i) = -a(i) <= s(i);
end
```

The objective integrates $\sum s(i)$. Minimizing it forces the solver to keep the torque magnitude small, subject to the dynamic constraints.

`acons_pos(i)` and `acons_neg(i)` implement $-s(i) \leq a(i) \leq s(i)$.

### E. Discretized Dynamics

```matlab
% Velocity dynamics
v_dynamics = optimconstr(N-1);
for i = 2:N
    v_dynamics(i-1) = ...
        theta_dot(i) == theta_dot(i-1) + dt * ((a(i-1)/I) 
                              - (b/I)*theta_dot(i-1) 
                              + (m_body*g*l_COM/I)*sin(theta(i-1)));
end

% Position (angle) dynamics
p_dynamics = optimconstr(N-1);
for i = 2:N
    p_dynamics(i-1) = ...
        theta(i) == theta(i-1) + dt*(theta_dot(i-1)+theta_dot(i))/2;
end
```

Velocity update ensures $\dot{\theta}$ at step $i$ follows the approximate Euler/trapezoidal integration.

Position update is using the trapezoidal rule:

$$\theta(i) = \theta(i-1) + \Delta t \frac{\dot{\theta}(i-1) + \dot{\theta}(i)}{2}.$$

### F. Rate-of-Torque-Development Constraints

```matlab
rtd_upper_constraints = optimconstr(N-2);
rtd_lower_constraints = optimconstr(N-2);
for i = 1:N-2
    rtd_upper_constraints(i) = (a(i+1)-a(i))/dt <= MRTD_df;
    rtd_lower_constraints(i) = (a(i+1)-a(i))/dt >= -MRTD_pf;
end
```

Restricts how quickly a(i) can change, i.e. bounding $\dot{a}$.

These constraints capture the real-world limit on how fast the ankle can develop torque.

### G. Initial and Final Conditions

```matlab
% Initial conditions
initialcons = [theta(1) == theta_0; 
               theta_dot(1) == theta_dot_0];

% Final conditions: angle must be target, velocity must be zero
finalcons = [theta(N) == theta_target; theta_dot(N) == 0];
```

Ensures the problem starts at $\theta_0$ and ends at $\theta_{target}$ with zero velocity.

### H. Angle Range Constraint

```matlab
max_angle = 30 * pi/180;
angle_range_upper = optimconstr(N);
angle_range_lower = optimconstr(N);
for i = 1:N
    angle_range_upper(i) = theta(i) <= max_angle;
    angle_range_lower(i) = theta(i) >= -max_angle;
end
```

Prevents unrealistic ankle angles, e.g. beyond ±30°.

### I. Combine All Constraints

```matlab
ankleProb.Constraints.acons_pos = acons_pos;
ankleProb.Constraints.acons_neg = acons_neg;
ankleProb.Constraints.v_dynamics = v_dynamics;
ankleProb.Constraints.p_dynamics = p_dynamics;
ankleProb.Constraints.rtd_upper_constraints = rtd_upper_constraints;
ankleProb.Constraints.rtd_lower_constraints = rtd_lower_constraints;
ankleProb.Constraints.initialcons = initialcons;
ankleProb.Constraints.finalcons = finalcons;
ankleProb.Constraints.angle_range_upper = angle_range_upper;
ankleProb.Constraints.angle_range_lower = angle_range_lower;
```

Assembles them under `ankleProb`.

### J. Solving and Post-Processing

```matlab
% Initial guess
x0.theta = linspace(theta_0, theta_target, N)';
x0.theta_dot = linspace(theta_dot_0, 0, N)';
x0.a = zeros(N-1, 1);
x0.s = ones(N-1, 1);

% Solve
options = optimoptions('fmincon', ... );
[sol,fval,exitflag,output] = solve(ankleProb,x0,'Options',options);

% Extract solution if successful
if exitflag > 0
    ...
else
    ...
end
```

Initial guess: A linear progression for $\theta$, $\theta\_dot$ and zero torque.

The solver uses `fmincon`. If it converges, the code extracts `theta_sol`, `theta_dot_sol`, `a_sol`, and logs or visualizes the results.

### K. Fallback and Plots

If the solver fails, the script uses a PD control routine as fallback.

The final portion plots $\theta$ vs. time, torque vs. time, etc., to visualize the optimal solution.

## Summary of Key Points

1. **Direct Collocation**: The script discretizes angle and velocity at each time step and forms constraints to mimic continuous dynamics.

2. **Torque as the Control**: `a(i)` is bounded and subject to rate-of-change constraints.

3. **Slack Variables**: $\sum s(i)$ in the objective effectively penalizes large $|a(i)|$.

4. **No Foot-Ground**: The code comments out ground-contact constraints (like friction/tipping), focusing purely on ankle torque limits and safe angle ranges.
