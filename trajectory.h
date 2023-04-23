#ifndef TRAJECTORY_H
#define TRAJECTORY_H
#include <iostream>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>
#include "mark.h"
#include <vector>

class Trajectory {
public:
    Trajectory(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance);

    void update_state_estimate(const Eigen::VectorXd& new_state_estimate);

    void update_covariance_matrix(const Eigen::MatrixXd& new_covariance_matrix);

    Eigen::VectorXd get_latest_state_estimate() const;

    Eigen::MatrixXd get_covariance_matrix() const;

    std::vector<Eigen::VectorXd> get_all_state_estimates() const;

    Eigen::MatrixXd get_H() const;

    double get_conditional_probability(const Mark& z_k) const;

    double get_prior_probability() const;

private:
    std::vector<Eigen::VectorXd> m_state_estimates;
    Eigen::MatrixXd m_covariance_matrix;
    Eigen::MatrixXd H;
};

#endif // TRAJECTORY_H
