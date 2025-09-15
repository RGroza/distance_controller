#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/utilities.hpp"
#include "rcpputils/scope_exit.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <iomanip>
#include <ostream>

using namespace std::chrono_literals;
using std::placeholders::_1;
using Odometry = nav_msgs::msg::Odometry;
using Twist = geometry_msgs::msg::Twist;
using Float32MultiArray = std_msgs::msg::Float32MultiArray;

constexpr float deriv_lpf_alpha = 0.2f;
constexpr float integ_limit = 0.5f;
constexpr float min_pos_error = 0.01f;

class DistanceController : public rclcpp::Node {
public:
  DistanceController(int scene_number) : Node("distance_controller"), scene_number_(scene_number) {
    odom_subscriber_ = this->create_subscription<Odometry>("odometry/filtered", 10,
                                                           std::bind(&DistanceController::odom_callback, this, _1));

    rclcpp::QoS qos_profile(rclcpp::KeepLast(10));
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

    cmd_vel_publisher_ = this->create_publisher<Twist>("/cmd_vel", qos_profile);
    data_publisher_ = this->create_publisher<Float32MultiArray>("/distance_controller_data", 10);

    controller_timer_ = this->create_wall_timer(10ms, std::bind(&DistanceController::execute_trajectory, this));

    goal_pos_.setZero();
    odom_pos_.setZero();
    odom_vel_.setZero();
    error_.setZero();
    error_dot_.setZero();
    integral_.setZero();
    twist_.setZero();
    last_twist_.setZero();

    // Assigning waypoints based on scene number
    switch (scene_number_) {
    case 1: // Simulation
      kp_ = 3.0f;
      ki_ = 1.0f;
      kd_ = 0.7f;
      kp_ang_ = 0.0f;
      max_twist_mag_ = 0.9f;
      max_accel_mag_ = 0.0f;
      waypoints_ = {{0.0f, 1.0f},   {0.0f, -1.0f}, {0.0f, -1.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
                    {-1.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f}};
      break;
    case 2: // CyberWorld
      kp_ = 3.0f;
      ki_ = 1.0f;
      kd_ = 0.5f;
      kp_ang_ = 4.0f;
      max_twist_mag_ = 0.3f;
      max_accel_mag_ = 1.0f;
      waypoints_ = {{0.90f, 0.0f}, {0.0f, -0.5f}, {0.0f, 0.5f}, {-0.90f, 0.0f}};
      break;
    default:
      RCLCPP_ERROR(this->get_logger(), "Invalid Scene Number: %d", scene_number_);
    }

    Kp_ << kp_, kp_;
    Ki_ << ki_, ki_;
    Kd_ << kd_, kd_;

    waypoint_idx_ = 0;

    cmd_vel_msg_ = Twist();
    data_msg_ = Float32MultiArray();
    data_msg_.data.resize(6);
  }

private:
  rclcpp::Subscription<Odometry>::SharedPtr odom_subscriber_;
  rclcpp::Publisher<Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Publisher<Float32MultiArray>::SharedPtr data_publisher_;
  rclcpp::TimerBase::SharedPtr controller_timer_;
  rclcpp::Time last_time_, init_goal_time_;
  Eigen::Vector2f goal_pos_, odom_pos_, odom_vel_;
  Eigen::Vector2f error_, error_dot_, integral_, twist_, last_twist_;
  Eigen::Vector2f Kp_, Ki_, Kd_;
  Eigen::Matrix2f Rot_;
  float kp_, ki_, kd_, kp_ang_;
  float max_twist_mag_, max_accel_mag_;
  float odom_yaw_, odom_init_yaw_;
  std::vector<Eigen::Vector2f> waypoints_;
  size_t waypoint_idx_;
  Twist cmd_vel_msg_;
  Float32MultiArray data_msg_;
  bool have_time_ = false;
  bool have_init_pos_ = false;
  bool goal_crossed_ = true;
  bool goal_finished_ = true;
  int scene_number_;

  void odom_callback(const Odometry::SharedPtr msg) {
    odom_pos_(0) = static_cast<float>(msg->pose.pose.position.x);
    odom_pos_(1) = static_cast<float>(msg->pose.pose.position.y);
    odom_vel_(0) = static_cast<float>(msg->twist.twist.linear.x);
    odom_vel_(1) = static_cast<float>(msg->twist.twist.linear.y);

    const auto q = msg->pose.pose.orientation;
    odom_yaw_ = static_cast<float>(std::atan2(2.0f * (q.w * q.z + q.x * q.y), 1.0f - 2.0f * (q.y * q.y + q.z * q.z)));

    if (!have_init_pos_) {
      goal_pos_ = odom_pos_;
      odom_init_yaw_ = odom_yaw_;
      Rot_ = Eigen::Rotation2D<float>(odom_init_yaw_).toRotationMatrix();
      have_init_pos_ = true;
      RCLCPP_INFO(this->get_logger(), "Initial odom pos:\t\t(%.2f, %.2f), yaw: %.2f", odom_pos_(0), odom_pos_(1),
                  odom_init_yaw_);
    }
  }

  void execute_trajectory() {
    if (!have_init_pos_)
      return;

    if (goal_finished_) {
      goal_finished_ = false;
      rclcpp::sleep_for(2000ms);

      if (waypoint_idx_ < waypoints_.size()) {
        goal_pos_ += Rot_ * waypoints_[waypoint_idx_];
      } else {
        rclcpp::shutdown();
        return;
      }

      RCLCPP_INFO(this->get_logger(), "Strafing to waypoint %zu:\t(%.2f, %.2f) --> (%.2f, %.2f)", waypoint_idx_ + 1,
                  waypoints_[waypoint_idx_][0], waypoints_[waypoint_idx_][1], goal_pos_(0), goal_pos_(1));

      waypoint_idx_++;
    }

    // Time step
    rclcpp::Time now = this->now();
    float dt = 0.01;
    if (have_time_)
      dt = static_cast<float>((now - last_time_).seconds());
    last_time_ = now;
    have_time_ = true;

    // Position error in the body frame
    error_ = Rot_.transpose() * (goal_pos_ - odom_pos_);

    // Derivative on measurement
    error_dot_ = -deriv_lpf_alpha * odom_vel_ + (1.0f - deriv_lpf_alpha) * error_dot_;

    // Integrate with clamp
    float error_norm = error_.norm();
    if (error_norm < 0.2f) {
      integral_ += error_ * dt;
      integral_ =
          integral_.cwiseMax(Eigen::Vector2f::Constant(-integ_limit)).cwiseMin(Eigen::Vector2f::Constant(integ_limit));
    } else {
      integral_ *= 0.95;
    }

    // PID control law
    twist_ = Kp_.cwiseProduct(error_) + Ki_.cwiseProduct(integral_) + Kd_.cwiseProduct(error_dot_);

    // Saturate twist magnitude
    float twist_norm = twist_.norm();
    if (twist_norm > max_twist_mag_)
      twist_ *= (max_twist_mag_ / twist_norm);

    // Deadzone near goal to stop tiny jitter and reset integral
    if (error_norm < min_pos_error) {
      integral_.setZero();

      if (!goal_crossed_) {
        init_goal_time_ = this->now();
        goal_crossed_ = true;
      }

      if (goal_crossed_ && this->now() - init_goal_time_ > rclcpp::Duration(250ms)) {
        twist_.setZero();
        goal_finished_ = true;
      }
    } else {
      goal_crossed_ = false;
    }

    // Apply vector magnitude acceleration limit on twist
    Eigen::Vector2f delta_twist = twist_ - last_twist_;
    if (max_accel_mag_ > 0.0f) {
      float max_delta_mag = max_accel_mag_ * dt;
      float delta_mag = delta_twist.norm();

      if (delta_mag > max_delta_mag)
        delta_twist *= (max_delta_mag / delta_mag);

      twist_ = last_twist_ + delta_twist;
    }
    last_twist_ = twist_;

    float yaw_error = odom_init_yaw_ - odom_yaw_;

    data_msg_.data[0] = error_norm;
    data_msg_.data[1] = odom_vel_.norm();
    data_msg_.data[2] = integral_.norm();
    data_msg_.data[3] = twist_.norm();
    data_msg_.data[4] = yaw_error;
    if (dt > 1e-2f)
      data_msg_.data[5] = delta_twist.norm() / dt;
    data_publisher_->publish(data_msg_);

    cmd_vel_msg_.linear.x = twist_(0);
    cmd_vel_msg_.linear.y = twist_(1);
    cmd_vel_msg_.angular.z = kp_ang_ * yaw_error;
    cmd_vel_publisher_->publish(cmd_vel_msg_);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  int scene_number = 1;
  if (argc > 1) {
    scene_number = std::atoi(argv[1]);
  }

  rclcpp::spin(std::make_shared<DistanceController>(scene_number));
  rclcpp::shutdown();
  return 0;
}