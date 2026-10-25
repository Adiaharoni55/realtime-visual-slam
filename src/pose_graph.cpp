#include "rtvm/pose_graph.hpp"

#include <stdexcept>
#include <string>

#include <ceres/ceres.h>

namespace rtvm {

namespace {

// Residual for one edge a -> b: the difference between the measured relative pose
// and the one predicted by the current estimates of a and b (6 numbers).
struct RelativePoseError {
    RelativePoseError(const Eigen::Vector3d& t_ab, const Eigen::Quaterniond& q_ab)
        : t_ab_measured_(t_ab), q_ab_measured_(q_ab) {}

    template <typename T>
    bool operator()(const T* p_a_ptr, const T* q_a_ptr,
                    const T* p_b_ptr, const T* q_b_ptr, T* residuals_ptr) const {
        // View the raw arrays as Eigen types (no copies).
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

        // Predicted relative pose: T_a^-1 * T_b.
        const Eigen::Quaternion<T> q_a_inv = q_a.conjugate();
        const Eigen::Quaternion<T> q_ab_est = q_a_inv * q_b;
        const Eigen::Matrix<T, 3, 1> p_ab_est = q_a_inv * (p_b - p_a);

        // Rotation error: what is left after undoing the estimate from the measurement.
        const Eigen::Quaternion<T> delta_q = q_ab_measured_.template cast<T>() * q_ab_est.conjugate();

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
        residuals.template head<3>() = p_ab_est - t_ab_measured_.template cast<T>();
        residuals.template tail<3>() = T(2.0) * delta_q.vec();   // ~ rotation error in radians
        return true;
    }

    static ceres::CostFunction* create(const Pose& measured) {
        return new ceres::AutoDiffCostFunction<RelativePoseError, 6, 3, 4, 3, 4>(
            new RelativePoseError(measured.translation(), Eigen::Quaterniond(measured.rotation())));
    }

    Eigen::Vector3d t_ab_measured_;
    Eigen::Quaterniond q_ab_measured_;
};

}  // namespace

void PoseGraph::checkId(int id) const {
    if (id < 0 || id >= numNodes()) {
        throw std::out_of_range("PoseGraph: invalid node id " + std::to_string(id)); 
    }
}

int PoseGraph::addNode(const Pose& initial) {
    // convert current pose to node 
    Eigen::Vector3d p = initial.translation();
    Eigen::Quaterniond q = Eigen::Quaterniond(initial.rotation());
    
    nodes_.push_back(Node{p, q});
    return static_cast<int>(nodes_.size()) - 1;
}

void PoseGraph::addEdge(int from, int to, const Pose& measured) {
    checkId(from);
    checkId(to);
    edges_.push_back(Edge{from, to, measured});
}

Pose PoseGraph::pose(int id) const {
    checkId(id);

    const Node& node = nodes_[id];
    Pose res = Pose::Identity();
    res.linear() = node.q.normalized().toRotationMatrix();
    res.translation() = node.p;

    return res;
}

void PoseGraph::optimize() {
    // Nothing to optimize without edges (and Ceres can't handle an empty problem).
    if (edges_.empty()) {
        return;
    }

    ceres::Problem problem;

    // One manifold object for all quaternions: it lets Ceres move a quaternion only in
    // ways that keep it a valid rotation (length 1). The problem takes ownership of it.
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

    for (const Edge& e: edges_) {
        Node& a = nodes_[e.from];
        Node& b = nodes_[e.to];

        problem.AddResidualBlock(
            RelativePoseError::create(e.measured),
            nullptr,   // no robust loss (plain squared error)
            a.p.data(), a.q.coeffs().data(),
            b.p.data(), b.q.coeffs().data());    
        
        problem.SetManifold(a.q.coeffs().data(), quaternion_manifold);
        problem.SetManifold(b.q.coeffs().data(), quaternion_manifold);

    }

    // Fix node 0: moving all poses together wouldn't change any relative motion,
    // so one pose must stay put to make the solution unique.
    // (Node 0 is in the problem only if some edge uses it.)
    Node& first = nodes_[0];
    if (problem.HasParameterBlock(first.p.data())) {
        problem.SetParameterBlockConstant(first.p.data());
        problem.SetParameterBlockConstant(first.q.coeffs().data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;   
    options.logging_type = ceres::SILENT;           
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

}  // namespace rtvm
