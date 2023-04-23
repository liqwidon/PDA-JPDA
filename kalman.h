#ifndef KALMAN_H
#define KALMAN_H
#include <iostream>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>
#include "mark.h"
#include <vector>
class KalmanFilter {
public:
    KalmanFilter(double delta_t, double phi_s, const Eigen::MatrixXd& R)
        : F_(3, 3), H_(1, 3), Q_(3, 3), R_(R) {

        // Initialize the state transition matrix F
        F_ << 1, delta_t, delta_t * delta_t / 2,
            0,       1,          delta_t,
            0,       0,               1;

        // Initialize the observation matrix H
        H_ << 1, 0, 0;

        // Initialize the process noise covariance matrix Q
        Q_ << delta_t * delta_t * delta_t * delta_t * delta_t / 20, delta_t * delta_t * delta_t * delta_t / 8, delta_t * delta_t * delta_t / 6,
            delta_t * delta_t * delta_t * delta_t / 8,            delta_t * delta_t * delta_t / 3,          delta_t * delta_t / 2,
            delta_t * delta_t * delta_t / 6,                      delta_t * delta_t / 2,                    delta_t;
        Q_ *= phi_s;
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
        // Predict the state at the next time step
        Eigen::VectorXd x_pred = F_ * x;
        Eigen::MatrixXd P_pred = F_ * P * F_.transpose() + Q_;
        return x_pred;
    }

    Eigen::MatrixXd predict_covariance(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
        // Predict the state covariance at the next time step
        Eigen::MatrixXd P_pred = F_ * P * F_.transpose() + Q_;
        return P_pred;
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(const Eigen::VectorXd& x_pred, const Eigen::MatrixXd& P_pred, const Eigen::VectorXd& z) {
        // Update the state and covariance based on the received measurement
        Eigen::MatrixXd S = H_ * P_pred * H_.transpose() + R_;
        Eigen::MatrixXd K = P_pred * H_.transpose() * S.inverse();

        Eigen::VectorXd x_updated = x_pred + K * (z - H_ * x_pred);
        Eigen::MatrixXd P_updated = (Eigen::MatrixXd::Identity(3, 3) - K * H_) * P_pred;

        return std::make_pair(x_updated, P_updated);
    }

    Eigen::MatrixXd get_H() {
        return H_;
    }

private:
    Eigen::MatrixXd F_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
};
#endif // KALMAN_H
