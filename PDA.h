#ifndef PDA_H
#define PDA_H
#include "trajectory.h"
#include "kalman.h"

class BayesianIdentifier {
public:
    BayesianIdentifier() {
        Eigen::MatrixXd R;
        R  << 1, 0, 0,
              0, 1, 0,
              0, 0, 1;
        m_kalman_filter() = KalmanFilter(5, 10, R);
    }
    ~BayesianIdentifier() {}

    void update_trajectories(const std::vector<Mark>& z_k) {
        // Perform the Bayesian identification algorithm
        auto Omega = hypothesis_generation(z_k);
//        auto x_k_tj = conditional_probabilities(z_k, Omega);
//        auto betta_tj = unconditional_probabilities(z_k, Omega);
//        auto x_k_t = resulting_score(x_k_tj, betta_tj);

        // Update trajectories based on x_k_t
    }

    std::vector<Trajectory> get_trajectories() const {
        return m_trajectories;
    }

    void calculate() {
        // Run the Bayesian identification algorithm for each time step
        std::vector<Mark> marks = {/* ... */};  // Create a vector of Mark objects with the input data
        std::vector<Trajectory> trajectories = {/* ... */};  // Create a vector of Trajectory objects with the input data

        // Create instances of the Marks and Trajectories classes
        Mark marks_instance(marks);
        Trajectory trajectories_instance(trajectories);

        // Generate association events
        auto association_events = hypothesis_generation(marks_instance, trajectories_instance);

        // Calculate conditional probabilities and unconditional probabilities
        std::vector<std::vector<Eigen::Vector3d>> all_conditional_estimates;
        std::vector<std::vector<double>> betta_tj;

        for (const auto& event : association_events) {
            all_conditional_estimates.push_back(conditional_probabilities(event, marks_instance, trajectories_instance));
            betta_tj.push_back(unconditional_probabilities(event, marks_instance, trajectories_instance));
        }

        // Calculate the resulting score
        auto resulting_estimates = resulting_score(all_conditional_estimates, betta_tj, /* strobing_matrix */);

        // Update trajectories with the new state estimates
        trajectories_instance.update_trajectories(resulting_estimates);

        // (Optional) Output trajectories or state estimates
        auto all_trajectories = trajectories_instance.get_trajectories();
    }

private:
    std::vector<Trajectory> m_trajectories;
    KalmanFilter m_kalman_filter();

    double chi_squared_inverse(double probability, int degrees_of_freedom) {
        boost::math::chi_squared_distribution<double> distribution(degrees_of_freedom);
        return boost::math::quantile(distribution, probability);
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> calculate_strobing_matrix(const std::vector<Mark>& z_k) {
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> strobing_matrix(z_k.size(), m_trajectories.size());

        double Pg = 0.99;  // Set the probability of strobing Pg
        int degrees_of_freedom = 3;  // Set the degrees of freedom for chi-squared distribution

        double threshold = chi_squared_inverse(Pg, degrees_of_freedom);

        for (size_t i = 0; i < z_k.size(); ++i) {
            for (size_t j = 0; j < m_trajectories.size(); ++j) {
                // Get the state estimate and covariance extrapolated to the time of measurement
                Eigen::VectorXd x_k_k_1 = m_kalman_filter().predict(m_trajectories[j].get_latest_state_estimate(), m_trajectories[j].get_covariance_matrix());
                Eigen::MatrixXd P_k_k_1 = m_kalman_filter().predict_covariance(m_trajectories[j].get_latest_state_estimate(), m_trajectories[j].get_covariance_matrix());

                // Calculate the residual vector and its covariance matrix
                Eigen::VectorXd residual = z_k[i].getPositioin() - m_kalman_filter().get_H() * x_k_k_1;
                Eigen::MatrixXd covariance = m_kalman_filter().get_H() * P_k_k_1 * m_kalman_filter().get_H().transpose() + z_k[i].getcovPosPos();

                double quadratic_form = residual.transpose() * covariance.inverse() * residual;

                bool isInGate = quadratic_form <= threshold;
                strobing_matrix(i, j) = isInGate;
            }
        }

        return strobing_matrix;
    }

    struct AssociationEvent {
        Eigen::MatrixXi association_matrix;
        std::vector<int> d_t;
        std::vector<int> tau_j;
        int pi;
        int get_trajectory_index(size_t mark_index) const {
            for (size_t traj_index = 0; traj_index < association_matrix.cols(); ++traj_index) {
                if (association_matrix(mark_index, traj_index) == 1) {
                    return static_cast<int>(traj_index);
                }
            }
            return -1; // Mark not associated with any trajectory in this event
        }

    };
    ///????????????????????????????
    std::vector<std::vector<int>> generate_association_combinations(const Eigen::MatrixXd& gating_matrix, int current_row, std::vector<int>& current_combination) {
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

    std::vector<AssociationEvent> generate_association_matrices(const Eigen::MatrixXd& gating_matrix) {
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
    ///????????????????????????????????????????
    // Algorithm stages
    std::vector<AssociationEvent> hypothesis_generation(const std::vector<Mark>& z_k) {
        // Generate hypotheses about the belonging of marks to trajectories
        // ...
        auto strobing_matrix = calculate_strobing_matrix(z_k);

        // Generate association matrices based on the strobing matrix
        auto association_matrices = generate_association_matrices(strobing_matrix);

        return association_matrices;
    }
   ///?????
    Eigen::VectorXd conditional_probabilities(const std::vector<Mark>& z_k, const std::vector<AssociationEvent>& association_events, const std::vector<Trajectory>& m_trajectories) {
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

            // Multiply by the a priori probability P(Tk|Z1:k-1) for each trajectory in the event
            for (int t = 0; t < event.d_t.size(); ++t) {
                if (event.d_t[t] == 1) {
                    probability *= m_trajectories[t].get_prior_probability();
                }
            }

            probabilities(i) = probability;
        }

        // Normalize the probabilities so they sum up to 1
        probabilities /= probabilities.sum();

        return probabilities;
    }

    Eigen::VectorXd unconditional_probabilities(const Eigen::VectorXd& conditional_probs, const std::vector<AssociationEvent>& association_events) {
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
    ///???????
    std::vector<std::vector<Eigen::Vector3d>> calculate_conditional_state_estimates(
        const std::vector<Mark>& z_k,
        const std::vector<AssociationEvent>& association_events/*,
        const std::vector<Trajectory>& m_trajectories*/) {

        std::vector<std::vector<Eigen::Vector3d>> conditional_estimates(association_events.size());

        for (size_t event_index = 0; event_index < association_events.size(); ++event_index) {
            const auto& event = association_events[event_index];
            conditional_estimates[event_index].resize(z_k.size(), Eigen::Vector3d::Zero());

            for (size_t mark_index = 0; mark_index < z_k.size(); ++mark_index) {
                int trajectory_index = event.get_trajectory_index(mark_index);

                if (trajectory_index >= 0) {
                    const auto& trajectory = m_trajectories[trajectory_index];

                    // Compute the Kalman gain K_k
                    Eigen::Matrix3d S_k = trajectory.get_H() * trajectory.get_covariance_matrix() * trajectory.get_H().transpose() + z_k[mark_index].getcovPosPos();
                    Eigen::Matrix3d K_k = trajectory.get_covariance_matrix() * trajectory.get_H().transpose() * S_k.inverse();

                    // Update the state estimate x_k|k using K_k and the residual vector
                    Eigen::Vector3d residual = z_k[mark_index].getPositioin() - trajectory.get_H() * trajectory.get_latest_state_estimate();
                    Eigen::Vector3d x_k_k = trajectory.get_latest_state_estimate() + K_k * residual;

                    // Store the conditional state estimate for this mark and event
                    conditional_estimates[event_index][mark_index] = x_k_k;
                }
            }
        }

        return conditional_estimates;
    }

    ///
    std::vector<Eigen::Vector3d> resulting_score(
        const std::vector<std::vector<Eigen::Vector3d>>& conditional_estimates,
        const std::vector<std::vector<double>>& betta_tj,
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
};


#endif // PDA_H
