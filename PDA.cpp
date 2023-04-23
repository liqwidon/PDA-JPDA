#include "PDA.h"

JPDA::JPDA() {
    Eigen::MatrixXd R;
    R  << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;
    m_kalman_filter() = KalmanFilter(5, 10, R);
}

void JPDA::update_trajectories(const std::vector<Eigen::Vector3d>& resulting_estimates, const std::vector<Eigen::Matrix3d>& resulting_covariances) {
    if (m_trajectories.size() != resulting_estimates.size() || m_trajectories.size() != resulting_covariances.size()) {
        throw std::runtime_error("The number of trajectories, resulting estimates, and resulting covariances must match.");
    }

    for (size_t i = 0; i < m_trajectories.size(); ++i) {
        m_trajectories[i].update_state_estimate(resulting_estimates[i]);
        m_trajectories[i].update_covariance_matrix(resulting_covariances[i]);
    }
}

std::vector<Trajectory> JPDA::get_trajectories() const {
    return m_trajectories;
}

void JPDA::calculate() {
    std::vector<Mark> marks = {/* ... */};
    std::vector<Trajectory> trajectories = {/* ... */};

    std::vector<Mark> marks_instance(marks);

    auto association_events = hypothesis_generation(marks_instance);

    std::vector<std::vector<Eigen::Vector3d>> all_conditional_estimates;
    Eigen::VectorXd betta_tj;
    all_conditional_estimates = calculate_conditional_state_estimates(marks_instance, association_events);
    betta_tj = unconditional_probabilities(conditional_probabilities(marks_instance, association_events), association_events);

    auto resulting_estimates = resulting_score(all_conditional_estimates, betta_tj, calculate_strobing_matrix(marks_instance));

//    m_trajectories.update_trajectories(resulting_estimates);

    auto all_trajectories = get_trajectories();
}

double JPDA::chi_squared_inverse(double probability, int degrees_of_freedom) {
    boost::math::chi_squared_distribution<double> distribution(degrees_of_freedom);
    return boost::math::quantile(distribution, probability);
}

Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> JPDA::calculate_strobing_matrix(const std::vector<Mark>& z_k) {
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> strobing_matrix(z_k.size(), m_trajectories.size());

    double Pg = 0.99;
    int degrees_of_freedom = 3;

    double threshold = chi_squared_inverse(Pg, degrees_of_freedom);

    for (size_t i = 0; i < z_k.size(); ++i) {
        for (size_t j = 0; j < m_trajectories.size(); ++j) {
            Eigen::VectorXd x_k_k_1 = m_kalman_filter().predict(m_trajectories[j].get_latest_state_estimate(), m_trajectories[j].get_covariance_matrix());
            Eigen::MatrixXd P_k_k_1 = m_kalman_filter().predict_covariance(m_trajectories[j].get_latest_state_estimate(), m_trajectories[j].get_covariance_matrix());

            Eigen::VectorXd residual = z_k[i].getPositioin() - m_kalman_filter().get_H() * x_k_k_1;
            Eigen::MatrixXd covariance = m_kalman_filter().get_H() * P_k_k_1 * m_kalman_filter().get_H().transpose() + z_k[i].getcovPosPos();

            double quadratic_form = residual.transpose() * covariance.inverse() * residual;

            bool isInGate = quadratic_form <= threshold;
            strobing_matrix(i, j) = isInGate;
        }
    }

    return strobing_matrix;
}


std::vector<std::vector<int>> JPDA::generate_association_combinations(const Eigen::MatrixXd& gating_matrix, int current_row, std::vector<int>& current_combination) {
    std::vector<std::vector<int>> result;

    if (current_row == gating_matrix.rows()) {
        result.push_back(current_combination);
        return result;
    }

    for (int col = 0; col < gating_matrix.cols(); ++col) {
        if (gating_matrix(current_row, col) == 1) {
            if (col == 0 || std::find(current_combination.begin(), current_combination.end(), col) == current_combination.end()) {
                current_combination[current_row] = col;
                auto child_combinations = generate_association_combinations(gating_matrix, current_row + 1, current_combination);
                result.insert(result.end(), child_combinations.begin(), child_combinations.end());
            }
        }
    }

    return result;
}

std::vector<JPDA::AssociationEvent> JPDA::generate_association_matrices(const Eigen::MatrixXd& gating_matrix) {
    int num_marks = gating_matrix.rows();
    int num_trajectories = gating_matrix.cols();
    std::vector<int> current_combination(num_marks, -1);
    auto association_combinations = generate_association_combinations(gating_matrix, 0, current_combination);

    std::vector<AssociationEvent> association_events;
    for (const auto& combination : association_combinations) {
        Eigen::MatrixXi association_matrix = Eigen::MatrixXi::Zero(num_marks, num_trajectories);
        std::vector<int> d_t(num_trajectories, 0);
        std::vector<int> tau_j(num_marks, 0);

        for (int i = 0; i < num_marks; ++i) {
            association_matrix(i, combination[i]) = 1;
            if (combination[i] > 0) {
                d_t[combination[i] - 1] = 1;
                tau_j[i] = 1;
            }
        }

        int pi = std::accumulate(tau_j.begin(), tau_j.end(), 0, [](int sum, int tau) { return sum + (1 - tau); });

        association_events.push_back({association_matrix, d_t, tau_j, pi});
    }

    return association_events;
}

std::vector<JPDA::AssociationEvent> JPDA::hypothesis_generation(const std::vector<Mark>& z_k) {
    auto strobing_matrix = calculate_strobing_matrix(z_k);
    auto association_matrices = generate_association_matrices(strobing_matrix);

    return association_matrices;
}


Eigen::VectorXd JPDA::conditional_probabilities(const std::vector<Mark>& z_k, const std::vector<JPDA::AssociationEvent>& association_events) {
    Eigen::VectorXd probabilities(association_events.size());

    for (size_t i = 0; i < association_events.size(); ++i) {
        const auto& event = association_events[i];
        double probability = 1.0;

        for (int j = 0; j < event.association_matrix.rows(); ++j) {
            for (int t = 0; t < event.association_matrix.cols(); ++t) {
                if (event.association_matrix(j, t) == 1) {
                    probability *= m_trajectories[t].get_conditional_probability(z_k[j]);
                }
            }
        }

        for (int t = 0; t < event.d_t.size(); ++t) {
            if (event.d_t[t] == 1) {
                probability *= m_trajectories[t].get_prior_probability();
            }
        }

        probabilities(i) = probability;
    }

    probabilities /= probabilities.sum();

    return probabilities;
}

Eigen::VectorXd JPDA::unconditional_probabilities(const Eigen::VectorXd& conditional_probs, const std::vector<JPDA::AssociationEvent>& association_events) {
    int num_marks = association_events.front().tau_j.size();
    int num_trajectories = association_events.front().d_t.size();
    Eigen::VectorXd betta_tj = Eigen::VectorXd::Zero(num_trajectories);

    for (size_t i = 0; i < association_events.size(); ++i) {
        const auto& event = association_events[i];
        double probability = conditional_probs(i);

        for (int t = 0; t < num_trajectories; ++t) {
            if (event.d_t[t] == 1) {
                betta_tj(t) += probability;
            }
        }
    }

    return betta_tj;
}

std::vector<std::vector<Eigen::Vector3d>> JPDA::calculate_conditional_state_estimates(
    const std::vector<Mark>& z_k,
    const std::vector<JPDA::AssociationEvent>& association_events) {

    std::vector<std::vector<Eigen::Vector3d>> conditional_estimates(association_events.size());

    for (size_t event_index = 0; event_index < association_events.size(); ++event_index) {
        const auto& event = association_events[event_index];
        conditional_estimates[event_index].resize(z_k.size(), Eigen::Vector3d::Zero());

        for (size_t mark_index = 0; mark_index < z_k.size(); ++mark_index) {
            int trajectory_index = event.get_trajectory_index(mark_index);

            if (trajectory_index >= 0) {
                const auto& trajectory = m_trajectories[trajectory_index];

                Eigen::Matrix3d S_k = trajectory.get_H() * trajectory.get_covariance_matrix() * trajectory.get_H().transpose() + z_k[mark_index].getcovPosPos();
                Eigen::Matrix3d K_k = trajectory.get_covariance_matrix() * trajectory.get_H().transpose() * S_k.inverse();

                Eigen::Vector3d residual = z_k[mark_index].getPositioin() - trajectory.get_H() * trajectory.get_latest_state_estimate();
                Eigen::Vector3d x_k_k = trajectory.get_latest_state_estimate() + K_k * residual;

                conditional_estimates[event_index][mark_index] = x_k_k;
            }
        }
    }

    return conditional_estimates;
}

std::vector<Eigen::Vector3d> JPDA::resulting_score(
    std::vector<std::vector<Eigen::Vector3d>>& conditional_estimates,
    Eigen::VectorXd& betta_tj,
    const Eigen::MatrixXd& strobing_matrix) {

    size_t num_trajectories = betta_tj.size();
    size_t num_marks = betta_tj.front().size();

    std::vector<Eigen::Vector3d> resulting_estimates(num_trajectories, Eigen::Vector3d::Zero());

    for (size_t t = 0; t < num_trajectories; ++t) {
        for (size_t j = 0; j < num_marks; ++j) {
            resulting_estimates[t] += conditional_estimates[t][j] * betta_tj[t][j] * strobing_matrix(j, t);
        }
    }

    return resulting_estimates;
}
