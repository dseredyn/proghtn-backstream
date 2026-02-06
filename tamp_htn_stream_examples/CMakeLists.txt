cmake_minimum_required(VERSION 3.16)
project(tamp_htn_stream_examples)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install multiple Python packages shipped by this ROS package
ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(tamp_htn_stream_velma)

# Install example configuration files
install(DIRECTORY examples
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/moveit_planning_test.py
  scripts/extract_old_tests.py
  scripts/test_basic_planning.py
  scripts/python_fcl_test.py
  scripts/pybullet_ray.py
  scripts/pybullet_collision.py
  scripts/test_velma_kinematics.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
