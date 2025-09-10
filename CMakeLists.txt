cmake_minimum_required(VERSION 3.8)
project(distance_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(distance_controller src/distance_controller.cpp)
ament_target_dependencies(distance_controller rclcpp std_msgs geometry_msgs nav_msgs Eigen3)
target_include_directories(distance_controller PUBLIC ${EIGEN3_INCLUDE_DIRS})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  distance_controller
  DESTINATION lib/${PROJECT_NAME})

ament_package()
