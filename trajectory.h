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
    Trajectory(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance)
        : m_covariance_matrix(initial_covariance) {
        m_state_estimates.push_back(initial_state);
        H << 1, 0, 0;
    }

    ///???
    void update_trajectories(const std::vector<Eigen::Vector3d>& resulting_estimates) {
        if (m_trajectories.size() != resulting_estimates.size()) {
            throw std::runtime_error("The number of trajectories and resulting estimates must match.");
        }

        for (size_t i = 0; i < m_trajectories.size(); ++i) {
            m_trajectories[i].update_state(resulting_estimates[i]);
        }
    }
    void update_state(const Eigen::Vector3d& new_state_estimate) {
        m_state_estimate = new_state_estimate;
        m_state_history.push_back(new_state_estimate);  // Store the new state estimate in the state history
    }
    ///???
    void update_state_estimate(const Eigen::VectorXd& new_state_estimate) {
        m_state_estimates.push_back(new_state_estimate);
    }

    void update_covariance_matrix(const Eigen::MatrixXd& new_covariance_matrix) {
        m_covariance_matrix = new_covariance_matrix;
    }

    Eigen::VectorXd get_latest_state_estimate() const {
        return m_state_estimates.back();
    }

    Eigen::MatrixXd get_covariance_matrix() const {
        return m_covariance_matrix;
    }

    std::vector<Eigen::VectorXd> get_all_state_estimates() const {
        return m_state_estimates;
    }

    Eigen::MatrixXd get_H() const{
        return H;
    }

    double get_conditional_probability(const Mark& z_k) const {
        // Calculate the residual vector v_k
        Eigen::Vector3d v_k = z_k.getPositioin() - H * m_state_estimates.back();

        // Calculate the residual covariance matrix S_k
        Eigen::Matrix3d S_k = H * m_covariance_matrix * H.transpose() + z_k.getcovPosPos();

        // Calculate the conditional probability using the multivariate Gaussian PDF formula
        double normalization = 1.0 / std::sqrt(S_k.determinant() * std::pow(2 * M_PI, 3));
        double exponent = -0.5 * v_k.transpose() * S_k.inverse() * v_k;
        double probability = normalization * std::exp(exponent);

        return probability;
    }

    double get_prior_probability() const {
        // Assuming that the prior probability is constant for all trajectories
        // The probability distribution will be normalized during the calculation of conditional probabilities
        return 1.0;
    }

private:
    std::vector<Eigen::VectorXd> m_state_estimates;
    Eigen::MatrixXd m_covariance_matrix;
    Eigen::Matrix<double, 1, 3> H;
};

#endif // TRAJECTORY_H
