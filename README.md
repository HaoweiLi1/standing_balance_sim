# Contact Aware Physics Simulation of Standing Balance - produced for Ozay Lab @ University of Michigan

This repository is meant to be used as a tool for simulating the simplified standing balance model seen in Figure I.1. While this model is simpler than the actual dynamics of a human, the two-link model allows one to characterize the stabilizable state space for human balancing.

Within this repository are three simulations

![mujoco_vs_theory_resize](https://github.com/celwell20/standing_balance_sim/assets/79417604/3b26e5d5-4d15-470c-8215-d78d28e7ac9a)

**Figure I.1.** MuJoCo Two-link Standing Balance Model (left) and Theoretical Two-link Standing Balance Model (right).

Table of contents
=================
<!--ts-->

   * [How to download the Human Standing Balance Repository](#Downloading-the-Simulation-Repository)
   * [How to run the Human Standing Balance Simulation](#Running-the-Human-Standing-Balance-Simulation)
     * [Simulation 1: Initial Humanoid](##Simulation-1:-Initial-Humanoid)
   * [Useful Links](#Useful-links)
   * [Experiments](#Experiments)
<!--te-->

## Downloading the Simulation Repository

1. Navigate to the directory of your choice in a new terminal.
2. In the directory, run the command `git clone https://github.com/celwell20/standing_balance_sim.git`
3. The Human Standing Balance Simulation should now be copied into the present terminal directory

To read from the `config.yaml` file please `pip` install the `pyyaml` libary: `python -m pip install pyyaml`

## Running the Human Standing Balance Simulation

There are three different Simulations one may execute in this repository.

### Simulation 1: Initial Humanoid

The initial humanoid uses the MuJoCo `<motor>` element to actuate the two-link model

### Simulation 2: Test Humanoid

### Simulation 3: Muscle-Tendon Humanoid

[MuJoCo Github Repo](https://github.com/google-deepmind/mujoco?tab=readme-ov-file)
**[MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/latest/python.html)**

## Useful links

1. [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom) - useful for understanding how to create MJCF XML Mujoco models.
2. [MuJoCo Types API Reference](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtsensor) - API reference for parameters related to MuJoCo model, data, visualization, and other simulation values.
      [MuJoCo full API Reference](https://mujoco.readthedocs.io/en/latest/APIreference/index.html) 
3. [Useful MuJoCo Jupyter Notebook Tutorial](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=Z6NDYJ8IOVt7) - tutorial for setting up MuJoCo models, simulation, and visualizers
4. [MuJoCo Modeling Reference](https://mujoco.readthedocs.io/en/stable/modeling.html) - describes the process of creating, compiling, controlling, and visualizaing MuJoCo models.
5. [MuJoCo Computation Reference](https://mujoco.readthedocs.io/en/latest/computation/index.html) - describes the math use to evaluate a MuJoCo simulation.
6. [MuJoCo Simulation Reference](https://mujoco.readthedocs.io/en/latest/programming/simulation.html#forward-dynamics)
7. [MuJoCo Visualization Reference](https://mujoco.readthedocs.io/en/latest/programming/visualization.html#rendering)



## Experiments

### Torque-Control Gravity Compensation:

https://github.com/celwell20/standing_balance_sim/assets/79417604/f031ce8b-8efb-48be-b92a-7127443c489e

Initial condition: 5 [degrees CCW from vertical]; Setpoint: 5 [degrees CCW from vertical]

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/65c185e4-adf5-42de-b526-46e0be942999)

**Figure E.1.** Joint Angle, Joint Velocity, and Front/Back Contact Forces during Gravity Compensation Simulation.

### Initial Condition != Setpoint Angle:

https://github.com/celwell20/standing_balance_sim/assets/79417604/3675282a-73d8-40ed-adff-e2c2407a6280

Initial condition: 10 [degrees CCW from vertical]; Setpoint: 5 [degrees CCW from vertical]

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/6d21722c-3071-40f3-8296-35888deb0f3e)

**Figure E.2.** Joint Angle, Joint Velocity, and Front/Back Contact Forces during Simulation.

### Perturbation Response:

Initial condition: 5 [degrees CCW from vertical]; Setpoint: 5 [degrees CCW from vertical]

https://github.com/celwell20/standing_balance_sim/assets/79417604/cdc0fe23-88d2-472e-ba57-103e65c04988

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/8d83e709-1281-4722-9ac1-62b554dac016)

**Figure E.3.** Joint Angle, Joint Velocity, Front/Back Contact Forces, and Perturbation Input during Simulation.

### Muscle-Tendon Bang-Bang Controller:

https://github.com/celwell20/standing_balance_sim/assets/79417604/dbbbcf62-1338-4276-9f76-8e3d9354ff0e




