#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/utilities.hpp"
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

// PID gains
constexpr float kp = 4.0f;
constexpr float ki = 1.0f;
constexpr float kd = 1.0f;

constexpr float deriv_lpf_alpha = 0.2f;
constexpr float integ_limit = 0.5f;

constexpr float max_twist_magnitude = 0.9f;
constexpr float min_pos_error = 0.01f;

class DistanceController : public rclcpp::Node {
public:
  DistanceController() : Node("distance_controller") {
    odom_subscriber_ = this->create_subscription<Odometry>("rosbot_xl_base_controller/odom", 10,
                                                           std::bind(&DistanceController::odom_callback, this, _1));

    cmd_vel_publisher_ = this->create_publisher<Twist>("/cmd_vel", 10);
    data_publisher_ = this->create_publisher<Float32MultiArray>("/distance_controller_data", 10);

    trajectory_timer_ = this->create_wall_timer(10ms, std::bind(&DistanceController::execute_trajectory, this));

    goal_pos_.setZero();
    init_pos_.setZero();
    odom_pos_.setZero();
    odom_vel_.setZero();
    odom_vel_prev_.setZero();
    error_.setZero();
    error_dot_.setZero();
    integral_.setZero();
    twist_.setZero();

    Kp_ << kp, kp;
    Ki_ << ki, ki;
    Kd_ << kd, kd;

    waypoints_b_ = {{{0.0f, 1.0f},
                     {0.0f, -1.0f},
                     {0.0f, -1.0f},
                     {0.0f, 1.0f},
                     {1.0f, 1.0f},
                     {-1.0f, -1.0f},
                     {1.0f, -1.0f},
                     {-1.0f, 1.0f},
                     {1.0f, 0.0f},
                     {-1.0f, 0.0f}}};

    // waypoints_b_ = {{{1.0f, 1.0f},
    //                  {-1.0f, -1.0f},
    //                  {1.0f, -1.0f},
    //                  {-1.0f, 1.0f}}};

    waypoint_idx_ = 0;

    cmd_vel_msg_ = Twist();
    data_msg_ = Float32MultiArray();
    data_msg_.data.resize(5);

    acc_norm_filtered_ = 0.0f;
  }

private:
  rclcpp::Subscription<Odometry>::SharedPtr odom_subscriber_;
  rclcpp::Publisher<Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Publisher<Float32MultiArray>::SharedPtr data_publisher_;
  rclcpp::TimerBase::SharedPtr trajectory_timer_;
  rclcpp::Time last_time_, init_goal_time_;
  Eigen::Vector2f goal_pos_, init_pos_, odom_pos_, odom_vel_, odom_vel_prev_;
  Eigen::Vector2f error_, error_dot_, integral_, twist_;
  Eigen::Vector2f Kp_, Ki_, Kd_;
  std::array<Eigen::Vector2f, 10> waypoints_b_;
  size_t waypoint_idx_;
  Twist cmd_vel_msg_;
  Float32MultiArray data_msg_;
  float acc_norm_filtered_;
  bool have_time_ = false;
  bool goal_crossed_ = true;
  bool goal_finished_ = true;

  void odom_callback(const Odometry::SharedPtr msg) {
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_vel_prev_ = odom_vel_;
    odom_vel_(0) = msg->twist.twist.linear.x;
    odom_vel_(1) = msg->twist.twist.linear.y;
  }

  void execute_trajectory() {
    if (goal_finished_) {
      goal_finished_ = false;
      rclcpp::sleep_for(2000ms);

      if (waypoint_idx_ < waypoints_b_.size()) {
        init_pos_ = goal_pos_;
        goal_pos_ += waypoints_b_[waypoint_idx_];
      } else {
        rclcpp::shutdown();
        return;
      }

      waypoint_idx_++;

      RCLCPP_INFO(this->get_logger(), "Strafing to waypoint %zu:\t(%.2f, %.2f)", waypoint_idx_,
                  waypoints_b_[waypoint_idx_][0], waypoints_b_[waypoint_idx_][1]);
    }

    // Time step
    rclcpp::Time now = this->now();
    float dt = 0.01;
    if (have_time_)
      dt = static_cast<float>((now - last_time_).seconds());
    last_time_ = now;
    have_time_ = true;

    // Position error
    error_ = goal_pos_ - odom_pos_;

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
    if (twist_norm > max_twist_magnitude)
      twist_ *= (max_twist_magnitude / twist_norm);

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

    cmd_vel_msg_.linear.x = twist_(0);
    cmd_vel_msg_.linear.y = twist_(1);
    cmd_vel_publisher_->publish(cmd_vel_msg_);

    // Approximate acceleration from odom twist measurements
    float vel_norm = odom_vel_.norm();
    float acc_norm = ((odom_vel_ - odom_vel_prev_) / dt).norm();
    float alpha = 0.02f;
    acc_norm_filtered_ = alpha * acc_norm + (1 - alpha) * acc_norm_filtered_;

    data_msg_.data[0] = error_norm;
    data_msg_.data[1] = vel_norm;
    data_msg_.data[2] = acc_norm_filtered_;
    data_msg_.data[3] = integral_(0);
    data_msg_.data[4] = integral_(1);
    data_publisher_->publish(data_msg_);
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DistanceController>());
  rclcpp::shutdown();
  return 0;
}