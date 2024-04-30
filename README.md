# MuJoCo Human Standing Balance Simulation - Ozay Lab @ University of Michigan

This repository is meant to be used as a tool for simulating the simplified standing balance model seen in Figure I.1. While this model is simpler than the actual dynamics of a human, the two-link model still allows one to characterize the region of state space, $[\theta_1, \dot{\theta}_1]$, that produces stablizable human balancing.

![mujoco_vs_theory_resize](https://github.com/celwell20/standing_balance_sim/assets/79417604/3b26e5d5-4d15-470c-8215-d78d28e7ac9a) \
**Figure I.1.** MuJoCo Two-link Standing Balance Model (left) w/ contact forces shown in red; Theoretical Two-link Model (right).

Table of contents
=================
<!--ts-->

   * [How to download the Human Standing Balance Repository](#Downloading-the-Simulation-Repository)
   * [Libraries to Install](#Libraries-to-Install)
   * [Useful Links](#Useful-links)
   * [How to use the XML Utility Script](#How-to-use-XML-Utility-Script)
   * [How to use the Configuration file](#How-to-use-the-Configuration-file)
   * [Python/MATLAB Plotter Scripts](#Python-and-MATLAB-Plotter-Scripts)
   * [How to build the two-link model in MuJoCo](#How-to-build-the-two-link-model-in-MuJoCo)
   * [How to run the Human Standing Balance Simulation](#Running-the-Human-Standing-Balance-Simulation)
     * [Initial Humanoid Simulation](#Initial-Humanoid-Simulation)
     * [Test Humanoid Simulation](#Test-Humanoid-Simulation)
     * [Final Humanoid Simulation](#Final-Humanoid-Simulation)
     * [Muscle-Tendon Humanoid Simulation](#Muscle-Tendon-Humanoid-Simulation)
   * [MuJoCo Simulation Class Architecture](#MuJoCo-Simulation-Class-Architecture)
     * [Controller Method](#Controller-Method)
     * [Perturbation Method](#Perturbation-Method)
     * [Run Method](#Run-Method)   
   * [Experiments](#Experiments)
<!--te-->

## Downloading the Simulation Repository

1. Navigate to the directory of your choice in a new terminal.
2. In the directory, run the command `git clone https://github.com/celwell20/standing_balance_sim.git`
3. The Human Standing Balance Simulation should now be copied into the present terminal directory

## Libraries to Install

Please `pip install` the following libraries if you do not already have them: <br>
1. MuJoCo library: `pip install mujoco`
2. `pyyaml` library: `python -m pip install pyyaml` <br>
3. `numpy`<br>
4. `threading` <br>
5. `xml` support library: `pip install xml-python` <br>
6. `imageio` <br>
7. `matplotlib` <br>

## Useful links

1. [MuJoCo Github Repo](https://github.com/google-deepmind/mujoco?tab=readme-ov-file) <br>
2. **[MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/latest/python.html)**
3. [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom) - useful for understanding how to create MJCF XML Mujoco models.
4. [MuJoCo Types API Reference](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtsensor) - API reference for parameters related to MuJoCo model, data, visualization, and other simulation values.
      [MuJoCo full API Reference](https://mujoco.readthedocs.io/en/latest/APIreference/index.html) 
5. [Useful MuJoCo Jupyter Notebook Tutorial](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=Z6NDYJ8IOVt7) - tutorial for setting up MuJoCo models, simulation, and visualizers
6. [MuJoCo Modeling Reference](https://mujoco.readthedocs.io/en/stable/modeling.html) - describes the process of creating, compiling, controlling, and visualizaing MuJoCo models.
7. [MuJoCo Computation Reference](https://mujoco.readthedocs.io/en/latest/computation/index.html) - describes the math use to evaluate a MuJoCo simulation.
8. [MuJoCo Simulation Reference](https://mujoco.readthedocs.io/en/latest/programming/simulation.html#forward-dynamics)
9. [MuJoCo Visualization Reference](https://mujoco.readthedocs.io/en/latest/programming/visualization.html#rendering)

## How to use XML Utility Script

The function of `xml_utilities.py` is to calculate the literature estimates of various parameters seen in Figure I.1's theoretical model, and subsequently apply them to the XML model we are interested in simulating with MuJoCo. 

We provide the *total mass*, $M_{total}$, and the *total height*, $H_{total}$, as arguments to the method `xml_utilities.calculate_kp_and_geom()` which returns all the desired human geometry and mass estimates, in addition to a *gravity compensation proprortional gain*, $K_P$, which is used as the proportional gain in the ankle motor's linear gravity compensation controller.

To apply these literature estimates to our XML model of interest, I used the `ElementTree` library (imported via `import xml.etree.ElementTree as ET`). This library allows us to iterate through different elements in the XML file, which makes it quite convenient when picking and choosing what values we want to edit without entering the XML file structure. The target XML's file path is used as the argument to parse it's XML tree: `ET.parse(xml_path)`. **You should configure the target XML path in the `config.yaml` file specific to your simulation**. We then extract the XML tree `root` (`root=tree.getroot()`) which we pass as the first argument into the `xml_utilities.set_geometry_params()` (in addition to the literature estimate dimensions). `set_geometry_params()` edits the XML model we are interested in modifying; once we have edited the XML we save it to a new XML file (to not override our original XML being modified). This modified XML file is then parsed by MuJoCo to get the `model` and `data` structs to be used in the simulation.

`config.yaml` original XML file parameter: `xml_path` <br>
`config.yaml` modified (typically modified to abide by literature estimate dimensions) XML file parameter: `lit_xml_file`

## How to use the Configuration file

The `config.yaml` file is where you may toggle on/off a number of visualization/simulation parameters, in addition to modifying numerical values that are used in the simulation. See Figure C.1 for a list of the available `config.yaml` parameters.

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/60f51bd0-602f-488d-a266-4073f2df3484) <br>
**Figure C.1.** Configurable parameters in `config.yaml` (As of 30th April 2024).

## Python and MATLAB Plotter Scripts

I have developed plotter tools in both Python and MATLAB. While the Python plotters are quicker to use, I would recommend using the MATLAB plotters as they are more updated and are used to produce the plots seen in the [Experiments](#Experiments) section (they look much nicer).

Available Python Plotter Methods:
1. `plot_3d_pose_trajectory` - plots the pose trajectory of the humanoid's center of mass as the simulation evovles
2. `plot_columns` - plots the first two columns of an arbitrary `numpy` array versus one another
3. `plot_two_columns` - plots the first two columns of two arbitrary `numpy` arrays. Both array data series's are displayed on the same plot
4. `plot_four_columns` - plots the first two columns of four arbitrary `numpy` arrays. All array data series are shown on the same plot.

I have configured the `initial_humanoid`, `test_humanoid`, `final_humanoid`, and `muscle_humanoid` simulations to automatically calculate the data of interest and save it to `numpy` arrays for processing in the Python plotters. **To turn the plotting visualization on/off, toggle the `plotter_flag` parameter in `config.yaml`.**

Available MATLAB Plotter Methods: <br>
1. `plotter.mlx` - creates a figure with four data series: <br>
   a. Joint position, $\theta$, versus time <br>
   b. Joint velocity, $\dot{\theta}$ versus time <br>
   c. Front and back contact forces, $F_{c,front}$ & $F_{c,back}$, versus time <br>
   d. Perturbation force, $f_{perturbation}$ <br>
   `plotter.mlx` also produces a plot of only the $f_{perturbation}$ data series. <br>
3. `plotter_3.mlx` - creates a figure with three data series: <br>
   a. Joint position, $\theta$, versus time <br>
   b. Joint velocity, $\dot{\theta}$ versus time <br>
   c. Front and back contact forces, $F_{c,front}$ & $F_{c,back}$, versus time <br>
   `plotter_3.mlx` also produces a plot of only the $f_{perturbation}$ data series. <br>

I have configured the `final_humanoid`, `test_humanoid`, and `muscle_humanoid` simulations to automatically calculate and save the data series' of interest to `.csv` files. So, you should be able to run those simulations, and then run the MATLAB plotters shortly after without modifying anything.

## How to build the two-link model in MuJoCo

Please refer to this presentation for details on the implementation of the two-link model in MuJoCo XML syntax: \
https://umich-my.sharepoint.com/:p:/g/personal/ctelwell_umich_edu/EaIz1NFO1XFEkzxwgJTZY6YBmNlhxBVn0IoqdKdbXYUzvA?e=Iq7vxv

## Running the Human Standing Balance Simulation

Within this repository are three simulations:
1. The first simulation, `initial_humanoid`, is the first revision of the human standing balance model built in MuJoco. It is fully functional for the purposes of representing the theoretical two-link model. <br>
2. The second simulation, `final_humanoid`, is the finished version of the humand standing balance model. The only difference between `final_humanoid` and `initial_humanoid` is that the reference frames of the links are rotated $180\degree$ about the worldframe $z$-axis (Figure R.1). This rotation is applied so that the reference frames of the `foot` and `long_link` MuJoCo `geom` elements align with the convention stated in the theoretical model.

![inital_versus_final_link_frames](https://github.com/celwell20/standing_balance_sim/assets/79417604/699ff693-4db3-4bdb-8308-65ea13b33858)
**Figure R.1.** Reference Frame Orientations for World Frame, Initial humanoid link frames, and Final humanoid link frames.

The `final_humanoid` simulation/directory is intended to be a location for completed MuJoCo models/simulations.

3. The third simulation, `test_model_motor`, is meant for testing and simulation development; you can think of it as a safe place to modify the XML model/MuJoCo simulation you're working on -- if things break, it doesn't matter, since there is a copy in the `final_humanoid` directory (assuming you're diligent about updating both directories). It has the same reference frame configuration as `final_humanoid` and as of 28th April, 2024 they are identical.

4. The fourth simulation, `muscle_humanoid`, is an exploration of creating a high-fidelity two-link humanoid via the MuJoCo muscle and tendon functionalities. This model has different foot dimensions than that of `initial_humanoid` or `final_humanoid`; in particular the foot is symmetrical. This change was implemented to make the moments exerted by the front and back muscles equivalent and ease the tuning process of the bang-bang controller that provides control inputs to the muscles.

### Initial Humanoid Simulation

The `initial_humanoid` simulation uses the MuJoCo `<motor>` element to actuate the two-link model; this model is dated compared to others in this repository. I left it in the repository because I didn't see a point in deleting it, but I highly recommend referencing `final_humanoid`, `test_humanoid`, and `muscle_humanoid` as they are much more recent. <br>

To run the `initial_humanoid` simulation: <br>
1. Open a terminal of your choice and enter the `standing_balance_sim` directory that is installed locally on your computer. <br>
2. Enter the proper simulation directory: `cd initial_humanoid` <br>
3. Run the command `python run_sim.py` in the terminal to run the simulation

### Final Humanoid Simulation

To run the `final_humanoid` simulation: <br>
1. Open a terminal of your choice and enter the `standing_balance_sim` directory that is installed locally on your computer. <br>
2. Enter the proper simulation directory: `cd final_humanoid` <br>
3. Run the command `python run_sim.py` in the terminal to run the simulation

### Test Humanoid Simulation

The `test_humanoid` simulation uses the `<motor>` element to actuate the ankle. It is meant for developing new simulations, without fear of breaking any polished simulations (store these in `final_humanoid` or create another "humanoid" simulation directory and store your good simulations there). 

To run the `test_humanoid` simulation: <br>
1. Open a terminal of your choice and enter the `standing_balance_sim` directory that is installed locally on your computer. <br>
2. Enter the proper simulation directory: `cd test_humanoid` <br>
3. Run the command `python run_sim.py` in the terminal to run the simulation

### Muscle-Tendon Humanoid Simulation

The `muscle_humanoid` simulation is an exploratory exercise to learn about `<muscle>` and `<tendon>` elements available to users in MuJoCo. These would allow one to create a high-fidelity -- and more biologically accurate -- standing balance model that could be used in future works. For details on the `muscle_humanoid` XML implementation and control, see slides 150 and onwards in the [documentation powerpoint](#How-to-build-the-two-link-model-in-MuJoCo).

To run the `muscle_humanoid` simulation: <br>
1. Open a terminal of your choice and enter the `standing_balance_sim` directory that is installed locally on your computer. <br>
2. Enter the proper simulation directory: `cd muscle_humanoid` <br>
3. Run the command `python run_sim.py` in the terminal to run the simulation

## MuJoCo Simulation Class Architecture
This section highlights how the various `humanoid` simulation classes are structured. There are three main methods, which are `controller`, `generate_large_impulse`, and `run`. Hopefully these are intuitively named, but for clarity, the `controller` class runs the control law during the simulation update, and assigns control inputs to the `<actuator>` elements in our model. The `generate_large_impulse` method generates quasi-impulse perturbations of varying magnitude, pulse width, and frequency; this method is called in a separate thread from the simulation thread. A future work for this tool would be to implement the perturbation in the actual simulation update, rather than in a separate thread. The two threads running simultaneously does not cause problems, but leads to perturbations that do not quite align with the user specified parameters in `config.yaml` (namely the perturbation pulse width). The `run` method loads the XML model, and runs the simulation.

### Controller Method
The `controller` method assigns control inputs to the `data.ctrl` MuJoCo struct, which subsequently applies these inputs to the various actuators in our MuJoCo XML model. To toggle the controller on/off, set the `controller_flag` parameter in the `config.yaml` file. <br>

In Figure CM.1 one may see the linear gravity compensation controller that is running in the `initial_humanoid`, `test_humanoid`, and `final_humanoid` simulations. <br>

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/aca2baf2-8311-47dc-abdf-2832663a33d9) <br>
**Figure CM.1.** Linear Gravity Compensation Controller.

Use the `data.sensordata` attribute to read from the sensors that exist in the XML model structure. `data.sensordata[0]` corresponds to the `jointpos` sensor that I have placed at the ankle joint, to measure the angular position, $\theta$, of the ankle joint. We compare `data.sensordata[0]` to the ankle joint setpoint, and set the controller output with the linear gravity compensation control law: $\tau = K_P \cdot errror$. The output torque is assigned to the `data.ctrl[0]` attribute, which corresponds to the torque input $\tau$ in Figure I.1's theoretical model; in the MuJoCo XML structure the ankle actuator is a `motor` MuJoCo element (Figure CM.2).

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/347d4c61-c4f0-4910-81a0-a4bde3e197f5) <br>
**Figure CM.2.** Ankle torque input implemented in MuJoCo XML model.

### Perturbation Method
The quasi-perturbation method, `generate_large_impulse`, is able to generate square-wave inputs that have a specific magnitude, pulse width, and frequency (Figure CM.3). To toggle the perturbation on/off, set the `perturbation_flag` parameter in the `config.yaml` file. <br>

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/845acbe6-4ee8-485f-9971-184960c6c3db) <br>
**Figure CM.3.** Quasi-impulse generator method. <br>

This method is running in a separate thread from the simulation, which is executed in the `run` method. <br>

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/2b4918e7-0232-4093-b921-b66b0fce80c5) <br>
**Figure CM.4.** Code responsible for starting the `generate_large_impulse` thread. <br>

It is important to note that since we started the perturbation thread (we do this via `perturbation_thread.start()`) we must also terminate the perturbation thread when the simulation is done running, which we do via `perturbation_thread.join()` (Figure CM.5). <br>

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/2f323408-74f1-471b-b536-823c5f2bbf76) <br>
**Figure CM.5.** This is the code in the `run` method that terminates the perturbation thread when the simulation is done running. <br>

In the perturbation generator, the perturbation signal is written to a `Queue` which is, subsequently processed in the `run` script to apply the perturbation force to the humanoid's *long link* center of mass (Figure CM.6).

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/7ccf3587-68f3-47e6-8299-1e4e17119a96) <br>
**Figure CM.6.** Applying the perturbation to the *long link* center of mass. <br>

We apply the perturbation to `data.xfrc_applied[2]` because the *long link* center of mass is concentrated in third XML `geom` element, which is the *long link* point mass representation.

### Run Method

`run` is arguably the most important method in the various simulation classes available in this repository. It is responsible for the following (in roughly sequential order): <br>
1. Reads from the `config.yaml` file and parses all the user-specified arguments <br>
2. Calls the `xml_utility` script to calculate literature estimtes and apply them to the XML model <br>
3. Parses the XML model and loads it into the MuJoCo backend <br>
4. Creates the GLFW and OpenGL windows for rendering; I don't really understand how the GLFW or OpenGL stuff works, but my code uses it to open a viewing window. I'll aim to post more updates that explain how this part of the simulation functions. <br>
5. Configures (a lot of) visualization options, adjusts simulation camera POV <br>
6. Sets joint position and velocity initial conditions <br>
7. Pass the `controller` method into the MuJoCo backend (toggled on/off by the user) <br>
8. Start the perturbtion thread (toggled on/off by the user) <br>

Now that the setup is complete, we may advance the simulation with time. A `while` loop is used to repeatedly call `mujoco.mj_step` until the user-specified `simend` time is met (configure this in `config.yaml`). A lot happens in each call of the `mujoco.mj_step` function; in particular: <br>
1. Rigid body dynamics are integrated with simulation ODE solver <br>
2. Contact solver constraints are updated <br>
3. Controller is called (one, or multiple times depending on the solver/simulation settings). I *think* in my simulation the controller is called once per time step, but I am not 100% sure. <br>

Outside of `mujoco.mj_step`, not too much happens in the `while` loop other than reading from the perturbation `Queue` and applying the perturbation forces to the humanoid if the `Queue` is not empty. Additionally, all the datalogging occurs in the simulation `while` loop.

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

### Muscle-Tendon Linear Gravity-Compensation Bang-Bang Controller:

https://github.com/celwell20/standing_balance_sim/assets/79417604/8e0e0bf8-1a1c-4283-a04a-ed9c36c0b648

![image](https://github.com/celwell20/standing_balance_sim/assets/79417604/7e2712c5-fee2-4fc9-9733-dd7a9185bb0c)
**Figure E.4.** Joint Angle, Joint Velocity, and Front/Back Contact Forces during `muscle_humanoid` gravity compensation Simulation.

