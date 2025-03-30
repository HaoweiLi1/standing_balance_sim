# Ankle Optimization with Exoskeleton Support

## 1. MATLAB Tutorial Learning Notes

### 1.1 Discretized Optimal Trajectory and Cone Programming

The MATLAB tutorial on "Discretized Optimal Trajectory" (often demonstrated with second-order cone programming) provides a framework for solving trajectory optimization problems by discretizing the dynamics in time. Key benefits:

- **Fixed-Time Discretization**: Splits the continuous problem into a finite set of time steps, making the optimization problem more tractable.

- **Decision Variables**: At each discrete time i, you define state variables (e.g., ankle angle θ, angular velocity θ̇) and control variables (e.g., torque a).

- **Constraints**: System dynamics become linear or nonlinear constraints, plus additional physical constraints (e.g., torque magnitude or foot–ground limits) can be enforced at each time step.

- **Norm-Based Cost**: The tutorial shows how to enforce |a(i)| ≤ s(i) and minimize ∑s(i) to keep the cost function linear in the solver (or to replicate a 1-norm penalty).

This approach differs from non-discretized optimization by avoiding noise in variable-step ODE solvers and allowing for more straightforward modeling of complex constraints.

### 1.2 Relationship to Our Ankle Model

For an ankle balance problem (modeled as an inverted pendulum or single-link system), this discretized approach provides several benefits:

- **States & Controls**: We discretize θ, θ̇, and the ankle torque a over N steps.

- **Dynamics as Equality Constraints**:
  ```
  θ(i) = θ(i−1) + Δt⋅(θ̇(i−1) + θ̇(i))/2,
  θ̇(i) = θ̇(i−1) + Δt [a(i−1)/I − b/I θ̇(i−1) + m*g*l_COM/I * sin(θ(i−1))].
  ```

- **Physical Bounds**:
  - Maximum torque magnitude and rate-of-torque-development (physiological limit).
  - Foot–ground constraints (to avoid tipping and slipping).

- **Cost Function**: Minimizing ∑|a(i)|Δt is akin to reducing total control effort.

This framework aligns perfectly with our need to optimize balance recovery trajectories while respecting both physiological and physical constraints.

## 2. Implementation Notes for ankle_optimal_simulation.m

### 2.1 Core Optimization Setup

In ankle_optimal_simulation.m, the basic direct-collocation approach is set up with:

- **System Parameters**: Mass, COM location, moment of inertia, damping, etc.

- **Decision Variables**: θ(i), θ̇(i), a(i), s(i).

- **Objective**:
  Minimize ∑s(i)⋅Δt, subject to |a(i)| ≤ s(i).

- **Dynamics Constraints**: Enforced via equality constraints in each discrete step.

- **Bounded Control**: 
  1. Torque magnitude ∈ [MT_pf, MT_df]
  2. Rate-of-torque-development constraints

```matlab
% Create optimization variables
theta = optimvar('theta', N, 1);
theta_dot = optimvar('theta_dot', N, 1);
a = optimvar('a', N-1, 1, 'LowerBound', MT_pf, 'UpperBound', MT_df);
s = optimvar('s', N-1, 1, 'LowerBound', 0);

% Define objective function: minimize control effort
ankleProb.Objective = sum(s) * dt;
```

### 2.2 Foot–Ground Contact Constraints

A major extension is the no-tipping and no-slipping bounds:

- **Tipping Constraints**:
  ```
  τ_ankle ∈ [−m*g*cos(θ)*l_heel, m*g*cos(θ)*l_toe],
  ```
  These ensure the center of pressure remains within the foot length.

- **Friction Constraints**:
  ```
  |τ_ankle| ≤ μ*m*g*min(l_toe, l_heel)
  ```
  This prevents the foot from slipping.

- **Combining**: The final net torque bound is the intersection of tipping and friction limits. In code, this is done with min() and max() to pick the stricter upper/lower torque bound at each state:

```matlab
% Apply most restrictive constraint (tipping vs. friction)
tau_upper = fcn2optimexpr(@(u1, u2) min(u1, u2), tau_upper_tipping, tau_friction_bound);
tau_lower = fcn2optimexpr(@(l1, l2) max(l1, l2), tau_lower_tipping, -tau_friction_bound);
```

### 2.3 Challenges and Solutions

- **Nonlinear Constraints**: Using fcn2optimexpr is key to handle expressions like cos(θ) in the problem-based framework.

- **Multiple Relations Issue**: MATLAB's problem-based approach doesn't support multiple relations per constraint element, requiring separate constraint arrays:
  ```matlab
  foot_upper_constraints = optimconstr(N-1);
  foot_lower_constraints = optimconstr(N-1);
  ```

- **Feasibility**: Tighter constraints (small final time, high friction, etc.) may cause infeasibility. Adjusting horizon or relaxing constraints can help if the solver fails.

- **Angle Range**: Adding explicit angle range constraints prevents extreme angles where the foot-ground model might break down:
  ```matlab
  angle_range_upper(i) = theta(i) <= max_angle;
  angle_range_lower(i) = theta(i) >= -max_angle;
  ```

- **Fallback Strategy**: A simple PD controller implementation provides a fallback if optimization fails.

## 3. Implementation Notes for ankle_exo_optimal_simulation.m

### 3.1 Incorporating Exoskeleton Torque

The exoskeleton torque τ_exo(θ,θ̇) is added algebraically to the human torque:

```
τ_ankle(i) = a(i) + τ_exo(θ(i), θ̇(i)).
```

Where:
- a(i) = human torque (decision variable)
- exoTorqueFun(...) = exo torque function, typically PD or gravity compensation

The key exoskeleton parameters include:
```matlab
exo_control_type = 'gravity_comp';  % Options: 'PD', 'gravity_comp', 'none'
exo_torque_limit = 50;              % Maximum exoskeleton torque (Nm)
Kp_exo = m_body * g * l_COM;        % Proportional gain for PD controller
Kd_exo = 0.3 * sqrt(m_body * l_COM^2 * Kp_exo); % Derivative gain
```

Control types implemented:
1. **PD Controller**: Provides assistance proportional to angle error and velocity
   ```matlab
   raw_torque = Kp * (theta_target - theta_val) - Kd * theta_dot_val;
   ```

2. **Gravity Compensation**: Counteracts the gravitational torque based on body angle
   ```matlab
   raw_torque = m * g * l * sin(theta_val);
   ```

3. **None**: No exoskeleton assistance (τ_exo = 0)

### 3.2 Code Modifications

- **Dynamics**:
  ```matlab
  exo_expr = fcn2optimexpr(exoTorque, theta(i-1), theta_dot(i-1));
  ankle_torque_expr = a(i-1) + exo_expr;
  ```
  Used in the velocity update constraints.

- **Foot–Ground Constraints**:
  ```matlab
  ankle_torque = a(i) + exo_expr;
  foot_upper_constraints(i) = ankle_torque <= tau_upper;
  foot_lower_constraints(i) = ankle_torque >= tau_lower;
  ```
  Ensures combined torque obeys the same no-slip/tip constraints.

- **Objective**: Still only penalizes the human torque ∑|a(i)|. The solver "exploits" exo torque if it helps reduce the user's effort (since the exo cost is zero in the default code).

### 3.3 Analyzing Exoskeleton Contribution

After solving, the implementation performs several analyses:

- **Torque Decomposition**:
  ```matlab
  % Calculate exoskeleton torque
  exo_torque_sol = zeros(N-1, 1);
  for i = 1:N-1
      exo_torque_sol(i) = exoTorque(theta_sol(i), theta_dot_sol(i));
  end
  
  % Calculate total ankle torque
  ankle_torque_sol = a_sol + exo_torque_sol;
  ```

- **Contribution Percentage**:
  ```matlab
  % Calculate percentage of exo contribution to total torque
  exo_percentage = zeros(N-1, 1);
  for i = 1:N-1
      if abs(ankle_torque_sol(i)) > 0.01  % Avoid division by near-zero
          exo_percentage(i) = 100 * exo_torque_sol(i) / ankle_torque_sol(i);
      else
          exo_percentage(i) = 0;
      end
  end
  ```

- **Visualization**: Dedicated plots showing human vs. exo contributions and their percentage over time.

- **Summary Statistics**:
  ```matlab
  average_exo_contribution = mean(abs(exo_torque_sol)) / mean(abs(ankle_torque_sol)) * 100;
  fprintf('Average exoskeleton torque contribution: %.1f%%\n', average_exo_contribution);
  ```

## 4. Key Insights and Future Directions

### 4.1 Physical Constraints

Including foot–ground contact constraints ensures physically valid solutions for balancing. Without these, the solver might find unrealistic torque strategies that would cause tipping in reality. The implementation properly accounts for:

- State-dependent torque limits based on foot geometry
- Friction constraints to prevent slipping
- Angle limits to maintain validity of the model

### 4.2 Exoskeleton Assistance

Because exo torque is included in the net dynamics and constraints, the solver's optimum for a(i) "knows" that exo help is available, thereby reducing human effort cost. The different control strategies offer flexibility:

- Gravity compensation works well for static or slow movements
- PD control can provide more dynamic assistance
- Parameters can be tuned to balance assistance vs. interference

### 4.3 Potential Extensions

1. **Penalize Exo Torque**: If you want to limit exo torque usage for energy or user-preference reasons, add it to the cost function:
   ```matlab
   ankleProb.Objective = sum(s_human) * dt + alpha * sum(s_exo) * dt;
   ```

2. **Multi-Phase or Free-Final-Time**: The direct-collocation approach can handle more complex tasks, e.g., stepping or variable final times.

3. **Rate Limits for Exo**: If your exoskeleton can't instantly change torque, add a second rate-of-torque-development constraint for τ_exo.

4. **Parameter Optimization**: Instead of fixed gains for the exoskeleton controller, optimize these parameters alongside the trajectory.

5. **User Adaptation Models**: Incorporate models of how humans adapt their control in response to exoskeleton assistance.

6. **Energy Efficiency**: Add energy consumption terms for both human and exoskeleton to the cost function.

By combining these elements—discretized dynamic constraints, torque bounds, foot–ground interaction constraints, and exoskeleton torque—we can develop physically plausible ankle balance control strategies that satisfy the constraints from the StabilizableRegions paper while leveraging exoskeleton support to minimize human effort.
