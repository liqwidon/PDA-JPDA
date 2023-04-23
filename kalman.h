#ifndef KALMAN_H
#define KALMAN_H

#include <iostream>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>
#include <vector>

class KalmanFilter {
public:
    KalmanFilter(double delta_t, double phi_s, const Eigen::MatrixXd& R);

    Eigen::VectorXd predict(const Eigen::VectorXd& x, const Eigen::MatrixXd& P);

    Eigen::MatrixXd predict_covariance(const Eigen::VectorXd& x, const Eigen::MatrixXd& P);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(const Eigen::VectorXd& x_pred, const Eigen::MatrixXd& P_pred, const Eigen::VectorXd& z);

    Eigen::MatrixXd get_H();

private:
    Eigen::MatrixXd F_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
};
#endif // KALMAN_H
