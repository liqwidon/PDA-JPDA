#ifndef PDA_H
#define PDA_H
#include "trajectory.h"
#include "kalman.h"

class JPDA {
public:
    JPDA();
    ~JPDA() {}

    void update_trajectories(const std::vector<Eigen::Vector3d>& resulting_estimates, const std::vector<Eigen::Matrix3d>& resulting_covariances);

    std::vector<Trajectory> get_trajectories() const;

    void calculate();

private:
    std::vector<Trajectory> m_trajectories;
    KalmanFilter m_kalman_filter();

    double chi_squared_inverse(double probability, int degrees_of_freedom);

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> calculate_strobing_matrix(const std::vector<Mark>& z_k);

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
            return -1;
        }
    };

    std::vector<std::vector<int>> generate_association_combinations(const Eigen::MatrixXd& gating_matrix, int current_row, std::vector<int>& current_combination);
    std::vector<AssociationEvent> generate_association_matrices(const Eigen::MatrixXd& gating_matrix);

    std::vector<AssociationEvent> hypothesis_generation(const std::vector<Mark>& z_k);

    Eigen::VectorXd conditional_probabilities(const std::vector<Mark>& z_k, const std::vector<AssociationEvent>& association_events);

    Eigen::VectorXd unconditional_probabilities(const Eigen::VectorXd& conditional_probs, const std::vector<AssociationEvent>& association_events);

    std::vector<std::vector<Eigen::Vector3d>> calculate_conditional_state_estimates(
        const std::vector<Mark>& z_k,
        const std::vector<AssociationEvent>& association_events);

    ///
    std::vector<Eigen::Vector3d> resulting_score(
        std::vector<std::vector<Eigen::Vector3d>>& conditional_estimates,
        Eigen::VectorXd& betta_tj,
        const Eigen::MatrixXd& strobing_matrix);
};


#endif // PDA_H
