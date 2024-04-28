# MuJoCo Human Standing Balance Simulation - Ozay Lab @ University of Michigan

This repository is meant to be used as a tool for simulating the simplified standing balance model seen in Figure I.1. While this model is simpler than the actual dynamics of a human, the two-link model still allows one to characterize the region of state space, $[\theta_1, \dot{\theta}_1]$, that produces stablizable human balancing.

![mujoco_vs_theory_resize](https://github.com/celwell20/standing_balance_sim/assets/79417604/3b26e5d5-4d15-470c-8215-d78d28e7ac9a) \
**Figure I.1.** MuJoCo Two-link Standing Balance Model (left) w/ contact forces shown in red; Theoretical Two-link Model (right).

Table of contents
=================
<!--ts-->

   * [How to download the Human Standing Balance Repository](#Downloading-the-Simulation-Repository)
   * [Things to `pip install`](#Things-to-`pip-install`)
   * [How to use `xml utilities`](How-to-use-XML-Utility-Script)
   * [How to use the Configuration file](#How-to-use-the-Configuration-file)
   * [Python/MATLAB Plotter Scripts](#Python/MATLAB-Plotter-Scripts)
   * [How to build the two-link model in MuJoCo](#How-to-build-the-two-link-model-in-MuJoCo)
   * [How to run the Human Standing Balance Simulation](#Running-the-Human-Standing-Balance-Simulation)
     * [Initial Humanoid Simulation](#Initial-Humanoid-Simulation)
     * [Test Humanoid Simulation](#Test-Humanoid-Simulation)
     * [Final Humanoid Simulation](#Final-Humanoid-Simulation)
     * [Muscle-Tendon Humanoid Simulation](#Muscle-Tendon-Humanoid-Simulation)
   * [Useful Links](#Useful-links)
   * [Experiments](#Experiments)
<!--te-->

## Downloading the Simulation Repository

1. Navigate to the directory of your choice in a new terminal.
2. In the directory, run the command `git clone https://github.com/celwell20/standing_balance_sim.git`
3. The Human Standing Balance Simulation should now be copied into the present terminal directory

## Things to `pip install`

To read from the `config.yaml` file please `pip` install the `pyyaml` libary: `python -m pip install pyyaml`

## How to use XML Utility Script

The function of `xml_utilities.py` is to calculated the literature estimates of various parameters seen in Figure I.1's theoretical model, and subsequently apply them to the XML model we are interested in simulating with MuJoCo. 

We provide the *total mass*, $M_{total}$, and the *total height*, $H_{total}$, as arguments to the method `xml_utilities.calculate_kp_and_geom()` which returns all the desired human geometry and mass estimates, in addition to a *gravity compensation proprortional gain*, $K_P$, which is used as the proportional gain in the ankle motor's linear gravity compensation controller.

To apply these literature estimates to our XML model of interest, I used the `ElementTree` library (imported via `import xml.etree.ElementTree as ET`). The target XML's file path is used as the argument to parse it's XML tree: `ET.parse(xml_path)`. **You should configure the target XML path in the `config.yaml` file specific to your simulation**.

## How to use the Configuration file

Write stuff about how the `config.yaml` file works

## Python/MATLAB Plotter Scripts

Write stuff about how to use the Python and MATLAB plotter tools

## How to build the two-link model in MuJoCo

Please refer to this presentation for details on the implementation of the two-link model in MuJoCo XML syntax: \
https://umich-my.sharepoint.com/:p:/g/personal/ctelwell_umich_edu/EaIz1NFO1XFEkzxwgJTZY6YBmNlhxBVn0IoqdKdbXYUzvA?e=Iq7vxv

## Running the Human Standing Balance Simulation

Within this repository are three simulations:
1. The first simulation, `initial_humanoid`, is the first revision of the human standing balance model built in MuJoco. It is fully functional for the purposes of representing the theoretical two-link model.
2. The second simulation, `final_humanoid`, is the finished version of the humand standing balance model. The only difference between `final_humanoid` and `initial_humanoid` is that the reference frames of the links are rotated $180\degree$ about the worldframe $z$-axis (Figure R.1)

![inital_versus_final_link_frames](https://github.com/celwell20/standing_balance_sim/assets/79417604/699ff693-4db3-4bdb-8308-65ea13b33858)
**Figure R.1.** Reference Frame Orientations for World Frame, Initial humanoid link frames, and Final humanoid link frames.

The `final_humanoid` simulation/directory is intended to be a location for completed MuJoCo models/simulations.

3. The third simulation, `test_model_motor`, is meant for testing and simulation development; you can think of it as a safe place to modify the XML model/MuJoCo simulation you're working on -- if things break, it doesn't matter, since there is a copy in the `final_humanoid` directory (assuming you're diligent about updating both directories). It has the same reference frame configuration as `final_humanoid` and as of 28th April, 2024 they are identical.

4. The fourth simulation, `muscle_humanoid`, is an exploration of creating a high-fidelity two-link humanoid via the MuJoCo muscle and tendon functionalities. This model has different foot dimensions than that of `initial_humanoid` or `final_humanoid`; in particular the foot is symmetrical. This change was implemented to make the moments exerted by the front and back muscles equivalent and ease the tuning process of the bang-bang controller that provides control inputs to the muscles.

### Initial Humanoid Simulation

The initial humanoid uses the MuJoCo `<motor>` element to actuate the two-link model; this model is not very up-to-date at this point. I left it in the repository because I didn't see a point in deleting it, but I highly recommend referencing `final_humanoid`, `test_humanoid`, and `muscle_humanoid` as they are much more recent. 

### Test Humanoid Simulation

### Final Humanoid Simulation

### Muscle-Tendon Humanoid Simulation



## Useful links

[MuJoCo Github Repo](https://github.com/google-deepmind/mujoco?tab=readme-ov-file)
**[MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/latest/python.html)**

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
Initial condition: $5\degree$ (CCW from vertical); Setpoint: $5\degree$ [CCW from vertical]

https://github.com/celwell20/standing_balance_sim/assets/79417604/f031ce8b-8efb-48be-b92a-7127443c489e

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/65c185e4-adf5-42de-b526-46e0be942999)\
**Figure E.1.** Joint Angle, Joint Velocity, and Front/Back Contact Forces during Gravity Compensation Simulation.

### Initial Condition $\neq$ Setpoint Angle:
Initial condition: $10\degree$ (CCW from vertical); Setpoint: $5\degree$ [CCW from vertical]

https://github.com/celwell20/standing_balance_sim/assets/79417604/3675282a-73d8-40ed-adff-e2c2407a6280

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/6d21722c-3071-40f3-8296-35888deb0f3e)\
**Figure E.2.** Joint Angle, Joint Velocity, and Front/Back Contact Forces during Simulation.

### Perturbation Response:

Initial condition: $5\degree$ (CCW from vertical); Setpoint: $5\degree$ [CCW from vertical]

https://github.com/celwell20/standing_balance_sim/assets/79417604/cdc0fe23-88d2-472e-ba57-103e65c04988

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/8d83e709-1281-4722-9ac1-62b554dac016)\
**Figure E.3.** Joint Angle, Joint Velocity, Front/Back Contact Forces, and Perturbation Input during Simulation.

### Muscle-Tendon Bang-Bang Controller:

https://github.com/celwell20/standing_balance_sim/assets/79417604/dbbbcf62-1338-4276-9f76-8e3d9354ff0e
