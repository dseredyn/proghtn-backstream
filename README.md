# proghtn-backstream

## Contents

tamp_htn_stream - a task and motion planner (TAMP) with the following properties:
* hierarchical task networks (HTN)
* PDDLStreams-like conditional generators:
  * PDF sampling
* progression search:
  * backtracking based on feedback information from streams

tamp_htn_stream_examples - an example planning domain for 2 a arm robot (WUT Velma)


## Dependencies

The planner was tested on the following setup:
* Ubuntu 24.04
* ROS 2 Jazzy
* Python 3.12
* PyBullet (installed in venv, with pip)


## Installation

Create workspace and venv environemnt for Python
```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ws_tamp/src
cd ~/ws_tamp/src

TODO
source venv/bin/activate

```

TODO: add venv configuration

Download and compile ROS 2 overlay workspace:
```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ws_tamp/src
cd ~/ws_tamp/src
git clone --branch v1.0.0 https://github.com/dseredyn/proghtn-backstream.git
git clone --branch v1.0.0 https://github.com/RCPRG-ros-pkg/velma_robot.git
cd ~/ws_tamp
python -m colcon build --symlink-install
```

## Run

Run MoveIt:
```bash
cd ~/ws_tamp
source install/setup.bash
ros2 launch velma_moveit_config planning_only.launch.py
```

Run planner:
```bash
cd ~/ws_tamp
source install/setup.bash
ros2 run tamp_htn_stream planner package://tamp_htn_stream_examples/examples/velma_domain.json  --offline-test package://tamp_htn_stream_examples/examples/velma_test.json
```
