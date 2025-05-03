# MATLAB Optimal Control for Ankle Exoskeleton Systems

This directory contains MATLAB models and implementations for optimal control strategies for ankle exoskeleton systems. This README integrates key insights from various implementation approaches.

## Table of Contents

1. [Enhanced Ankle Exoskeleton Control System](#1-enhanced-ankle-exoskeleton-control-system)
2. [Ankle Optimization with Exoskeleton Support](#2-ankle-optimization-with-exoskeleton-support)
3. [3D Lifted System for Optimal Ankle Control](#3-3d-lifted-system-for-optimal-ankle-control)
4. [Direct Collocation without Ground Constraints](#4-direct-collocation-without-ground-constraints)

---

## 1. Enhanced Ankle Exoskeleton Control System

### 1.1 Exoskeleton PD Controller Enhancements

#### Dynamic Gain Calculation

The exoskeleton PD controller has been significantly improved by implementing dynamic gain calculation based on biomechanical principles from the academic literature. Instead of using static PD gains, the controller now calculates appropriate gains based on the user's anthropometric data:

```python
# Calculate dynamic gains
kp = K_p  # Already calculated as m_body * g * l_COM
kd = 0.3 * np.sqrt(m_body * l_COM**2 * kp)
```

This approach ensures the controller is properly tuned to each user's specific body parameters rather than using one-size-fits-all values.

#### Integration with Human LQR Controller

The exoskeleton controller is now better integrated with the human controller, allowing for more natural coordination between human and exoskeleton torques:

```python
# For LQR, also pass exo configuration if exo is enabled
if exo_type != "None":
    human_params['exo_config'] = exo_config
    
    # Get the pre-calculated values from xml_utilities for dynamic gains
    if exo_type == "PD" and exo_config.get('pd_params', {}).get('use_dynamic_gains', False):
        # Pass dynamic gains to LQR controller
        human_params['dynamic_gains'] = {
            'kp': kp,
            'kd': kd
        }
```

#### Modified System Dynamics

The human LQR controller now accounts for the exoskeleton's effect on system dynamics by modifying the system matrices:

```python
# Modify system matrices to account for exoskeleton PD control
# The exo PD controller effectively changes the system dynamics:
# 1. Adds damping (b + Kd)
# 2. Changes effective stiffness (mgl - Kp)
A_exo = np.array([
    [0, 1],
    [(self.m*self.g*self.l - exo_kp)/self.I, -(self.b + exo_kd)/self.I]
])
```

#### Gear Ratio Scaling

The controller now properly scales gains by the exoskeleton gear ratio to match the control space:

```python
# Scale gains by the exoskeleton gear ratio to match the control space
exo_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")
if exo_actuator_id >= 0:
    exo_gear_ratio = self.model.actuator_gear[exo_actuator_id][0]
    exo_kp = exo_kp / exo_gear_ratio  # Scale to control space
    exo_kd = exo_kd / exo_gear_ratio  # Scale to control space
```

### 1.2 Modified Human LQR Controller for Exoskeleton Interaction

#### Mathematical Derivation

The human LQR controller has been enhanced to account for the presence of the exoskeleton. This adaptation is critical because the exoskeleton alters the dynamics of the combined human-exoskeleton system.

For the human-only system, the state-space representation follows a standard inverted pendulum model:
```
ẋ = Ax + Bu
```

When the exoskeleton is active, its PD controller applies torque based on:
```
τ_exo = -Kp·θ - Kd·θ̇
```

This exoskeleton torque changes the effective dynamics of the system. The new state equation becomes:
```
ẋ = A_exo x + Bu
```

The LQR control law for both cases is:
```
u = -Kx
```

### 1.3 Implementation of MATLAB-Generated Torque Curves in MuJoCo

The simulation now includes a `HumanPrecomputedController` class that allows for importing and using externally generated torque trajectories, particularly those calculated in MATLAB.

#### Torque Trajectory Comparison Analysis

The comparison between MATLAB-generated, MuJoCo-recorded, and actually applied torque trajectories reveals interesting findings:

1. **Early Phase Agreement**: All three trajectories closely match until approximately 0.1175s.
2. **Divergence After Initial Phase**: After 0.1175s, the MuJoCo-recorded trajectory begins to diverge.
3. **Recovery Phase Differences**: The trajectories show different recovery characteristics.
4. **Final State Discrepancy**: The final torque values differ between MATLAB and MuJoCo implementations.

The primary reason for discrepancy appears to be model complexity differences:
- The MATLAB model uses a simplified 1-DOF inverted pendulum assumption
- MuJoCo uses a more complex 4-DOF model with foot dynamics and ground interaction

## 2. Ankle Optimization with Exoskeleton Support

### 2.1 MATLAB Tutorial Learning Notes

#### Discretized Optimal Trajectory and Cone Programming

The MATLAB tutorial on "Discretized Optimal Trajectory" provides a framework for solving trajectory optimization problems by discretizing the dynamics in time. Key benefits:

- **Fixed-Time Discretization**: Splits the continuous problem into a finite set of time steps.
- **Decision Variables**: At each discrete time i, you define state variables and control variables.
- **Constraints**: System dynamics become linear or nonlinear constraints.
- **Norm-Based Cost**: Enforces |a(i)| ≤ s(i) and minimizes ∑s(i) to keep the cost function linear.

#### Relationship to Our Ankle Model

For an ankle balance problem (modeled as an inverted pendulum), this discretized approach provides several benefits:

- **States & Controls**: We discretize θ, θ̇, and the ankle torque a over N steps.
- **Dynamics as Equality Constraints**: Enforced at each time step.
- **Physical Bounds**: Maximum torque magnitude and foot–ground constraints.
- **Cost Function**: Minimizing ∑|a(i)|Δt reduces total control effort.

### 2.2 Core Optimization Setup

In the ankle_optimal_simulation.m implementation:

- **System Parameters**: Mass, COM location, moment of inertia, damping, etc.
- **Decision Variables**: θ(i), θ̇(i), a(i), s(i).
- **Objective**: Minimize ∑s(i)⋅Δt, subject to |a(i)| ≤ s(i).
- **Dynamics Constraints**: Enforced via equality constraints.
- **Bounded Control**: Torque magnitude and rate-of-torque-development constraints.

### 2.3 Foot–Ground Contact Constraints

A major extension is the no-tipping and no-slipping bounds:

- **Tipping Constraints**: τ_ankle ∈ [−m*g*cos(θ)*l_heel, m*g*cos(θ)*l_toe]
- **Friction Constraints**: |τ_ankle| ≤ μ*m*g*min(l_toe, l_heel)
- **Combining**: The final net torque bound is the intersection of tipping and friction limits.

### 2.4 Incorporating Exoskeleton Torque

The exoskeleton torque τ_exo(θ,θ̇) is added algebraically to the human torque:

```
τ_ankle(i) = a(i) + τ_exo(θ(i), θ̇(i)).
```

Control types implemented:
1. **PD Controller**: Assistance proportional to angle error and velocity
2. **Gravity Compensation**: Counteracts the gravitational torque
3. **None**: No exoskeleton assistance

### 2.5 Analyzing Exoskeleton Contribution

After solving, the implementation performs several analyses:
- **Torque Decomposition**: Separate human and exo contributions
- **Contribution Percentage**: Calculate how much the exo is contributing
- **Visualization**: Plot human vs. exo contributions over time
- **Summary Statistics**: Calculate average exoskeleton contribution

### 2.6 Potential Extensions

1. **Penalize Exo Torque**: Add exo torque cost to the objective function
2. **Multi-Phase or Free-Final-Time**: Handle more complex tasks
3. **Rate Limits for Exo**: Add rate-of-torque-development constraint for τ_exo
4. **Parameter Optimization**: Optimize controller parameters alongside the trajectory
5. **User Adaptation Models**: Model how humans adapt to exoskeleton assistance
6. **Energy Efficiency**: Add energy consumption terms to the cost function

## 3. 3D Lifted System for Optimal Ankle Control

### 3.1 Theoretical Basis: The "3D Lifted System"

Unlike simpler problems where torque is directly the control, the 3D lifted approach treats torque τ itself as part of the system state:

#### State and Control Variables

**States** at each discrete time step i:
- θ(i) – ankle angle
- θ_dot(i) – ankle angular velocity
- τ(i) – ankle torque (lifted into the state)

**Control** at each step i:
- u(i) – the rate of change of torque (τ̇)

#### Objective Function

The script penalizes the sum of squared torque-rate values:
```
min ∑(i=1 to N-1) u(i)^2 Δt
```

By penalizing u², the solver promotes smooth and gradual torque changes.

#### Dynamics Constraints

Three sets of updates are enforced from step i-1 to step i:

**Angle** (trapezoidal rule):
```
θ(i) = θ(i-1) + Δt [θ_dot(i-1) + θ_dot(i)]/2
```

**Angular Velocity**:
```
θ_dot(i) = θ_dot(i-1) + Δt [τ(i-1)/I - b/I θ_dot(i-1) + mgl_COM/I sin(θ(i-1))]
```

**Torque as a state** (Euler update):
```
τ(i) = τ(i-1) + Δt · u(i-1)
```

#### Torque and Torque-Rate Bounds
- Torque τ is bounded τ ∈ [MT_pf, MT_df]
- The control u (torque-rate) is bounded by ±MRTD

#### Initial and Final Conditions
- Initial: θ(1) = θ₀, θ_dot(1) = θ_dot₀, τ(1) = 0
- Final: θ(N) = θ_target, θ_dot(N) = 0, τ(N) unconstrained

### 3.2 Key Differences Compared to Direct Torque Control

- By including τ as a state, the approach has a straightforward handle on torque and its derivative
- The objective ∑u² punishes rapid torque changes rather than torque magnitude itself
- Rate-of-torque constraints become trivial: |u(i)| ≤ MRTD
- The solution can more smoothly shape the torque profile

## 4. Direct Collocation without Ground Constraints

### 4.1 Theoretical Basis for the Implementation

The ankle_optimal_no_foot_ground.m implements a direct collocation approach to optimal control:

#### State and Control Variables

**State:**
- θ(i): the ankle angle at step i
- θ_dot(i): the ankle angular velocity at step i

**Control:**
- a(i): the torque at each discrete step

Additionally, a "slack" variable s(i) is used to capture the absolute value of a(i) in the objective function.

#### Objective Function

The code minimizes:
```
∑ s(i) Δt
```
where |a(i)| ≤ s(i). This is effectively an L₁-type penalty on torque amplitude.

#### Dynamics and Constraints

**Discretized EoM:** Uses the trapezoidal rule for the ankle angle θ and Euler integration for velocity.

**Torque and Rate-of-Torque-Development Limits:**
- LowerBound ≤ a(i) ≤ UpperBound
- Difference constraint (a(i+1) - a(i))/Δt ≤ MRTD_df, ≥ -MRTD_pf

**Boundary Conditions:**
- θ(1) = θ₀, θ_dot(1) = θ_dot₀
- θ(N) = θ_target, θ_dot(N) = 0

### 4.2 Why Use Slack Variables?

Normally, one cannot directly write min ∑|a(i)| because absolute values are non-differentiable. The typical trick is to add a slack s(i) and constraints -s(i) ≤ a(i) ≤ s(i). Minimizing ∑s(i) then encourages |a(i)| to be minimal.

### 4.3 Key Points

1. **Direct Collocation**: Discretizes angle and velocity at each time step with constraints mimicking continuous dynamics.
2. **Torque as the Control**: a(i) is bounded and subject to rate-of-change constraints.
3. **Slack Variables**: ∑s(i) in the objective effectively penalizes large |a(i)|.
4. **No Foot-Ground**: Focus is purely on ankle torque limits and safe angle ranges, without ground-contact constraints.

---

## Summary and Integration Points

These four approaches to optimal control for ankle exoskeletons demonstrate different aspects of the same underlying problem:

1. **Note_precomputed**: Focuses on practical implementation and controller integration, connecting theoretical models from MATLAB with the MuJoCo physics simulation.

2. **Note_optimal**: Emphasizes the optimization framework, including foot-ground interaction constraints and different exoskeleton control modes.

3. **Optimal_3D**: Introduces a "lifted" approach where torque itself becomes a state variable, allowing for smoother torque profiles and easier handling of torque-rate constraints.

4. **Optimal_no_ground_constraints**: Presents a simpler direct collocation implementation that focuses on basic dynamics without ground interaction.

Together, these approaches provide a comprehensive toolkit for designing, optimizing, and implementing controllers for ankle exoskeletons under various conditions and constraints.