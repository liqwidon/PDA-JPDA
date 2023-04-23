#include "trajectory.h"

Trajectory::Trajectory(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance)
    : m_covariance_matrix(initial_covariance), H(1, 3) {
    m_state_estimates.push_back(initial_state);
    H << 1, 0, 0;
}

void Trajectory::update_state_estimate(const Eigen::VectorXd& new_state_estimate) {
    m_state_estimates.push_back(new_state_estimate);
}

void Trajectory::update_covariance_matrix(const Eigen::MatrixXd& new_covariance_matrix) {
    m_covariance_matrix = new_covariance_matrix;
}

Eigen::VectorXd Trajectory::get_latest_state_estimate() const {
    return m_state_estimates.back();
}

Eigen::MatrixXd Trajectory::get_covariance_matrix() const {
    return m_covariance_matrix;
}

std::vector<Eigen::VectorXd> Trajectory::get_all_state_estimates() const {
    return m_state_estimates;
}

Eigen::MatrixXd Trajectory::get_H() const{
    return H;
}

double Trajectory::get_conditional_probability(const Mark& z_k) const {
    Eigen::Vector3d v_k = z_k.getPositioin() - H * m_state_estimates.back();

    Eigen::Matrix3d S_k = H * m_covariance_matrix * H.transpose() + z_k.getcovPosPos();

    double normalization = 1.0 / std::sqrt(S_k.determinant() * std::pow(2 * M_PI, 3));
    double exponent = -0.5 * v_k.transpose() * S_k.inverse() * v_k;
    double probability = normalization * std::exp(exponent);

    return probability;
}

double Trajectory::get_prior_probability() const {
    return 1.0;
}
